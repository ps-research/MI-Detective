"""
MI User — Experiment Runner

Experiments:
  1. Feature Discovery — contrastive activation differentials per user property
  2. Circuit Tracing — attribution graphs for top user-model features
  3. Causal Validation — ablate user-model features, measure behavioral change
  4. Formation Timeline — when do user-model features activate in the prompt?

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/run_experiments.py --quick
  CUDA_VISIBLE_DEVICES=0 python experiments/run_experiments.py
"""

import sys
import os
import argparse
import time
import json
import numpy as np

sys.path.insert(0, "/workspace/Gemma-Scope-2-Study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt, GEMMA3_1B_NUM_LAYERS
from src.hooks import gather_clt_activations
from src.attribution import build_attribution_graph, prune_graph, compute_graph_metrics, save_graph
from src.interventions import ablate_clt_features, print_intervention_result

from prompts.prompts_mi_user import (
    PROPERTIES, get_all_property_ids, get_property_info,
    get_property_pairs, format_chat_prompt,
)


# ============================================================
# Experiment 1: Feature Discovery
# ============================================================

def run_feature_discovery(model, tokenizer, clt, property_id, device="cuda"):
    """
    Find features that activate differently on high vs low variants.

    For each contrastive pair:
      1. Run CLT on both high and low prompts
      2. At the last token of the user turn, extract all active features
      3. Compute activation differential: delta_a = a_high - a_low
      4. Rank features by |delta_a| averaged across pairs

    Returns dict with top differential features per property.
    """
    info = get_property_info(property_id)
    pairs = get_property_pairs(property_id)
    print(f"\n  Property: {info['name']} ({info['high_label']} vs {info['low_label']})")

    # Collect features for each pair
    all_differentials = {}  # (layer, feat_idx) -> list of delta_a values

    for pair_idx, (high_text, low_text) in enumerate(pairs):
        print(f"    Pair {pair_idx+1}/{len(pairs)}...", end="", flush=True)

        for variant, text in [("high", high_text), ("low", low_text)]:
            inputs = tokenizer.encode(text, return_tensors="pt",
                                      add_special_tokens=True).to(device)
            clt_inputs, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
            if next(clt.parameters()).dtype == torch.float16:
                clt_inputs = clt_inputs.half()

            features = clt.encode(clt_inputs)

            # Use last position features (just before model turn)
            last_pos = features.shape[0] - 1

            if variant == "high":
                feats_high = features[last_pos].float().cpu()  # (26, 10080)
            else:
                feats_low = features[last_pos].float().cpu()

        # Compute differential
        delta = feats_high - feats_low  # (26, 10080)

        # Record nonzero differentials
        for layer in range(delta.shape[0]):
            nonzero = delta[layer].nonzero(as_tuple=True)[0]
            for idx in nonzero:
                key = (layer, idx.item())
                if key not in all_differentials:
                    all_differentials[key] = []
                all_differentials[key].append(delta[layer, idx].item())

        print(f" {len(all_differentials)} candidate features")

    # Rank by mean |differential| across pairs
    ranked = []
    for (layer, feat_idx), deltas in all_differentials.items():
        mean_abs = np.mean([abs(d) for d in deltas])
        mean_signed = np.mean(deltas)
        consistency = np.std(deltas) / (mean_abs + 1e-8)  # lower = more consistent
        ranked.append({
            "layer": layer,
            "feature_idx": feat_idx,
            "mean_abs_differential": mean_abs,
            "mean_signed_differential": mean_signed,
            "consistency": consistency,
            "n_pairs": len(deltas),
            "all_deltas": deltas,
        })

    ranked.sort(key=lambda x: x["mean_abs_differential"], reverse=True)

    # Print top 15
    print(f"\n    Top 15 differentially active features:")
    print(f"    {'Layer':>5s} {'Feat':>6s} {'Mean|Δ|':>8s} {'MeanΔ':>8s} {'Consist':>8s}")
    for r in ranked[:15]:
        print(f"    L{r['layer']:>3d} f{r['feature_idx']:>5d} {r['mean_abs_differential']:>8.1f} "
              f"{r['mean_signed_differential']:>+8.1f} {r['consistency']:>8.2f}")

    return {
        "property_id": property_id,
        "property_name": info["name"],
        "top_features": ranked[:50],  # keep top 50 for downstream analysis
        "total_candidates": len(ranked),
    }


# ============================================================
# Experiment 2: Circuit Tracing
# ============================================================

def run_circuit_tracing(model, tokenizer, clt, property_id, top_features,
                        device="cuda"):
    """
    Trace attribution graphs on the best contrastive pair for this property.
    Focus on how user-model features connect to output behavior.
    """
    info = get_property_info(property_id)
    pairs = get_property_pairs(property_id)

    # Use first pair (strongest example)
    high_text, low_text = pairs[0]
    high_raw = PROPERTIES[property_id]["pairs"][0]["high"]
    low_raw = PROPERTIES[property_id]["pairs"][0]["low"]

    print(f"\n  Circuit tracing for {info['name']}:")
    print(f"    HIGH: {high_raw[:60]}...")
    print(f"    LOW:  {low_raw[:60]}...")

    results = {}
    for variant, text, label in [("high", high_text, info["high_label"]),
                                  ("low", low_text, info["low_label"])]:
        print(f"\n    Building attribution graph for {label}...")
        try:
            graph = build_attribution_graph(
                model, clt, tokenizer, text,
                top_k_output_tokens=10,
                min_ff_edge_weight=50.0,
                min_fl_edge_weight=5.0,
            )
            pruned = prune_graph(graph, top_k_edges_per_node=5,
                                max_feature_nodes=40, min_edge_weight=10.0)
            metrics = compute_graph_metrics(pruned)

            results[variant] = {
                "label": label,
                "metrics": metrics,
                "n_nodes": metrics.get("num_feature_nodes", 0),
                "n_ff_edges": metrics.get("feature_to_feature_edges", 0),
                "n_fl_edges": metrics.get("feature_to_logit_edges", 0),
                "avg_path_length": metrics.get("avg_path_length", 0),
                "layer_distribution": metrics.get("layer_distribution", {}),
            }

            # Check which top differential features appear in the graph
            top_feat_keys = {(f["layer"], f["feature_idx"]) for f in top_features[:20]}
            graph_feat_keys = set()
            for nid, node in pruned.feature_nodes.items():
                graph_feat_keys.add((node.layer, node.feature_idx))

            overlap = top_feat_keys & graph_feat_keys
            results[variant]["user_feature_overlap"] = len(overlap)
            results[variant]["overlap_features"] = list(overlap)

            print(f"      Nodes: {results[variant]['n_nodes']}, "
                  f"FF edges: {results[variant]['n_ff_edges']}, "
                  f"Path length: {results[variant]['avg_path_length']:.1f}")
            print(f"      User-model features in graph: {len(overlap)}/{len(top_feat_keys)}")

            # Save graph
            save_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "outputs")
            os.makedirs(save_dir, exist_ok=True)
            save_graph(pruned, f"{save_dir}/circuit_{property_id}_{variant}.json")

        except Exception as e:
            print(f"      FAILED: {e}")
            results[variant] = {"error": str(e)}

    return results


# ============================================================
# Experiment 3: Causal Validation
# ============================================================

def run_causal_validation(model, tokenizer, clt, property_id, top_features,
                          device="cuda", max_new_tokens=80):
    """
    Ablate top user-model features and measure behavioral change.
    """
    info = get_property_info(property_id)
    pairs = get_property_pairs(property_id)
    high_text, low_text = pairs[0]

    # Get top 3 features to ablate
    feature_specs = [(f["layer"], f["feature_idx"]) for f in top_features[:3]]
    print(f"\n  Ablating top 3 features: {feature_specs}")

    inputs_high = tokenizer.encode(high_text, return_tensors="pt",
                                    add_special_tokens=True).to(device)

    # Generate without ablation
    with torch.no_grad():
        clean_output = model.generate(input_ids=inputs_high, max_new_tokens=max_new_tokens,
                                       do_sample=False)
    gen_clean = tokenizer.decode(clean_output[0], skip_special_tokens=False)
    # Extract model response
    if "<start_of_turn>model" in gen_clean:
        gen_clean = gen_clean.split("<start_of_turn>model")[-1].strip()
    gen_clean = gen_clean.split("<end_of_turn>")[0].strip()

    # Ablate and measure logit change
    result = ablate_clt_features(model, clt, tokenizer, inputs_high, feature_specs)

    print(f"\n    Clean generation: {gen_clean[:150]}...")
    print(f"    Delta loss from ablation: {result.delta_loss:+.4f}")
    print(f"    Top predictions clean: {result.top_tokens_clean[:5]}")
    print(f"    Top predictions ablated: {result.top_tokens_intervened[:5]}")

    return {
        "property_id": property_id,
        "ablated_features": feature_specs,
        "generation_clean": gen_clean,
        "delta_loss": result.delta_loss,
        "top_clean": result.top_tokens_clean[:5],
        "top_ablated": result.top_tokens_intervened[:5],
    }


# ============================================================
# Experiment 4: Formation Timeline
# ============================================================

def run_formation_timeline(model, tokenizer, clt, property_id, top_features,
                            device="cuda"):
    """
    Track when user-model features first activate during the prompt.
    Feed the prompt token by token, recording feature activation at each step.
    """
    info = get_property_info(property_id)
    pairs = get_property_pairs(property_id)

    # Track top 5 features
    track_features = [(f["layer"], f["feature_idx"]) for f in top_features[:5]]
    print(f"\n  Tracking {len(track_features)} features across prompt positions")

    results = {}
    for variant, text, label in [("high", pairs[0][0], info["high_label"]),
                                  ("low", pairs[0][1], info["low_label"])]:
        inputs = tokenizer.encode(text, return_tensors="pt",
                                  add_special_tokens=True).to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())
        seq_len = inputs.shape[1]

        # Run full sequence and get features at each position
        clt_inputs, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
        if next(clt.parameters()).dtype == torch.float16:
            clt_inputs = clt_inputs.half()

        all_features = clt.encode(clt_inputs)  # (seq_len, 26, 10080)

        # Extract activation timeline for tracked features
        timelines = {}
        for layer, feat_idx in track_features:
            key = f"L{layer}/f{feat_idx}"
            acts = all_features[:, layer, feat_idx].float().cpu().tolist()
            timelines[key] = acts

            # Find first activation position
            first_pos = -1
            for pos, act in enumerate(acts):
                if act > 0:
                    first_pos = pos
                    break

            if first_pos >= 0 and first_pos < len(tokens):
                print(f"    {label} | {key}: first fires at pos {first_pos} "
                      f"('{tokens[first_pos]}'), act={acts[first_pos]:.1f}")
            else:
                print(f"    {label} | {key}: never fires")

        results[variant] = {
            "label": label,
            "tokens": tokens,
            "timelines": timelines,
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MI User Experiments")
    parser.add_argument("--quick", action="store_true", help="Run on 1 property only")
    parser.add_argument("--property", type=str, default=None,
                        help="Run on specific property (e.g., P1_expertise)")
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE = os.path.join(ROOT, "Gemma-Scope-2-Study", "cache")
    if not os.path.exists(CACHE):
        CACHE = "/workspace/Gemma-Scope-2-Study/cache"
    OUT = os.path.join(ROOT, "outputs")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("MI USER — EXPERIMENTS")
    print("=" * 70)

    # Load IT model (user modeling is chat behavior)
    model, tokenizer = load_gemma3_1b("it", device="cuda")

    # Load CLT-IT
    print("\nLoading CLT-IT (affine)...")
    clt = load_clt(
        width="262k", l0="big", affine=True, variant="it",
        device="cuda", half_precision=True, cache_dir=CACHE,
    )

    # Select properties to run
    if args.property:
        property_ids = [args.property]
    elif args.quick:
        property_ids = ["P1_expertise"]
    else:
        property_ids = get_all_property_ids()

    all_results = {}

    for prop_id in property_ids:
        print(f"\n{'='*70}")
        print(f"PROPERTY: {get_property_info(prop_id)['name']}")
        print(f"{'='*70}")

        # Exp 1: Feature Discovery
        print(f"\n--- Experiment 1: Feature Discovery ---")
        discovery = run_feature_discovery(model, tokenizer, clt, prop_id)

        # Exp 2: Circuit Tracing
        print(f"\n--- Experiment 2: Circuit Tracing ---")
        circuits = run_circuit_tracing(
            model, tokenizer, clt, prop_id,
            discovery["top_features"])

        # Exp 3: Causal Validation
        print(f"\n--- Experiment 3: Causal Validation ---")
        validation = run_causal_validation(
            model, tokenizer, clt, prop_id,
            discovery["top_features"])

        # Exp 4: Formation Timeline
        print(f"\n--- Experiment 4: Formation Timeline ---")
        timeline = run_formation_timeline(
            model, tokenizer, clt, prop_id,
            discovery["top_features"])

        all_results[prop_id] = {
            "discovery": {
                "total_candidates": discovery["total_candidates"],
                "top_features": discovery["top_features"][:20],
            },
            "circuits": circuits,
            "validation": validation,
            "timeline": {k: {kk: vv for kk, vv in v.items() if kk != "timelines"}
                        for k, v in timeline.items()},
        }

    # Save all results
    with open(f"{OUT}/mi_user_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {OUT}/mi_user_results.json")

    print(f"\n{'='*70}")
    print(f"GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
    print("MI USER EXPERIMENTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

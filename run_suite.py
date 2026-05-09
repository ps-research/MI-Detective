#!/usr/bin/env python3
"""
MI DETECTIVE — COMPLETE EXPERIMENT SUITE
=========================================
"MI for Investigating Mysterious Behaviours in Language Models"

4 Experiments across 12 case studies (pick best 5-6 for paper):
  1. Feature Fingerprinting — identify behavior-specific features
  2. Circuit Tracing — attribution graphs for trigger vs control
  3. Intervention Validation — ablate fingerprint features, check behavior change
  4. Verdict Classification — classify each behavior mechanistically

Usage:
  python run_suite.py --quick              # 2 cases (~25 min)
  python run_suite.py --case C2            # specific case
  python run_suite.py --cases C1,C2,C9,C10 # selected cases
  python run_suite.py                      # all 12 cases (~4 hr)

Requires: Gemma 3 1B IT, CLT-IT (affine)
GPU Memory: ~17 GB
"""

import sys, os, json, time, argparse, traceback, logging
from datetime import datetime

WORKSPACE = "/workspace"
INFRA = os.path.join(WORKSPACE, "Gemma-Scope-2-Study")
PROJECT = os.path.join(WORKSPACE, "MI-Detective")
sys.path.insert(0, INFRA)
sys.path.insert(0, PROJECT)

import torch
import numpy as np
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt, GEMMA3_1B_NUM_LAYERS
from src.hooks import gather_clt_activations
from src.attribution import build_attribution_graph, prune_graph, compute_graph_metrics, save_graph
from src.generation import generate_with_ablation, compare_generations, compute_behavioral_metrics

from prompts.prompts_mi_detective import (
    CASES, get_all_case_ids, get_case_info, get_case_prompts,
)

CACHE = os.path.join(INFRA, "cache")
OUT = os.path.join(PROJECT, "outputs")
os.makedirs(OUT, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUT, "experiment.log"), mode="w"),
    ],
)
log = logging.getLogger("mi_detective")


def save_results(data, filename):
    path = os.path.join(OUT, filename)
    data["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")


# ============================================================
# EXPERIMENT 1: Feature Fingerprinting
# ============================================================

def run_exp1_fingerprint(model, tokenizer, clt, case_ids):
    """Identify features unique to each behavior vs its control."""
    log.info("=" * 70)
    log.info("EXPERIMENT 1: FEATURE FINGERPRINTING")
    log.info("=" * 70)

    results = {"cases": {}}

    for case_id in case_ids:
        info = get_case_info(case_id)
        prompts = get_case_prompts(case_id)

        if info["is_multi_turn"]:
            log.info(f"\n  {info['name']} — multi-turn, handled separately")
            continue

        log.info(f"\n  CASE: {info['name']} (expected: {info['expected_verdict']})")

        all_diffs = {}

        for pi, prompt_data in enumerate(prompts):
            trigger_text = prompt_data["trigger"]
            control_text = prompt_data["control"]
            log.info(f"    Prompt {pi+1}/{len(prompts)}: {prompt_data['notes']}")

            try:
                # Encode trigger
                inp_t = tokenizer.encode(trigger_text, return_tensors="pt",
                                         add_special_tokens=True).to("cuda")
                clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inp_t)
                if next(clt.parameters()).dtype == torch.float16:
                    clt_in = clt_in.half()
                feats_trigger = clt.encode(clt_in)[-1].float().cpu()

                # Encode control
                inp_c = tokenizer.encode(control_text, return_tensors="pt",
                                          add_special_tokens=True).to("cuda")
                clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inp_c)
                if next(clt.parameters()).dtype == torch.float16:
                    clt_in = clt_in.half()
                feats_control = clt.encode(clt_in)[-1].float().cpu()

                delta = feats_trigger - feats_control

                for layer in range(delta.shape[0]):
                    nz = (delta[layer].abs() > 10).nonzero(as_tuple=True)[0]
                    for idx in nz:
                        key = (layer, idx.item())
                        if key not in all_diffs:
                            all_diffs[key] = []
                        all_diffs[key].append(delta[layer, idx].item())

                log.info(f"      {len(all_diffs)} candidates")

            except Exception as e:
                log.error(f"    Prompt {pi} failed: {e}")
                log.error(traceback.format_exc())

        # Rank
        ranked = []
        for (layer, feat_idx), deltas in all_diffs.items():
            ranked.append({
                "layer": layer, "feature_idx": feat_idx,
                "mean_abs_diff": float(np.mean([abs(d) for d in deltas])),
                "mean_signed_diff": float(np.mean(deltas)),
                "n_prompts": len(deltas),
                "all_deltas": [float(d) for d in deltas],
            })
        ranked.sort(key=lambda x: x["mean_abs_diff"], reverse=True)

        # Layer histogram
        layer_hist = {}
        for f in ranked[:30]:
            l = f["layer"]
            layer_hist[l] = layer_hist.get(l, 0) + 1

        results["cases"][case_id] = {
            "name": info["name"],
            "expected_verdict": info["expected_verdict"],
            "n_prompts": len(prompts),
            "total_candidates": len(ranked),
            "top_features": ranked[:50],
            "layer_distribution_top30": layer_hist,
        }

        log.info(f"    Top 5:")
        for f in ranked[:5]:
            log.info(f"      L{f['layer']}/f{f['feature_idx']}: |Δ|={f['mean_abs_diff']:.0f}")

    save_results(results, "exp1_fingerprints.json")
    return results


# ============================================================
# EXPERIMENT 2: Circuit Tracing
# ============================================================

def run_exp2_circuits(model, tokenizer, clt, case_ids, fingerprint_results):
    """Build attribution graphs for trigger vs control."""
    log.info("=" * 70)
    log.info("EXPERIMENT 2: CIRCUIT TRACING")
    log.info("=" * 70)

    results = {"circuits": {}}

    for case_id in case_ids:
        info = get_case_info(case_id)
        if info["is_multi_turn"]:
            continue

        prompts = get_case_prompts(case_id)
        if not prompts:
            continue

        top_feats = fingerprint_results.get("cases", {}).get(case_id, {}).get("top_features", [])
        prompt_data = prompts[0]  # Use first prompt

        log.info(f"\n  {info['name']}: tracing circuits...")

        case_results = {}
        for variant, text, label in [
            ("trigger", prompt_data["trigger"], "Trigger"),
            ("control", prompt_data["control"], "Control"),
        ]:
            try:
                graph = build_attribution_graph(
                    model, clt, tokenizer, text,
                    top_k_output_tokens=10,
                    min_ff_edge_weight=50.0, min_fl_edge_weight=5.0)
                pruned = prune_graph(graph, top_k_edges_per_node=5,
                                    max_feature_nodes=40, min_edge_weight=10.0)
                metrics = compute_graph_metrics(pruned)

                top_keys = {(f["layer"], f["feature_idx"]) for f in top_feats[:20]}
                graph_keys = {(n.layer, n.feature_idx) for n in pruned.feature_nodes.values()}
                overlap = top_keys & graph_keys

                case_results[variant] = {
                    "label": label,
                    "n_nodes": metrics.get("num_feature_nodes", 0),
                    "n_ff_edges": metrics.get("feature_to_feature_edges", 0),
                    "n_fl_edges": metrics.get("feature_to_logit_edges", 0),
                    "avg_path": metrics.get("avg_path_length", 0),
                    "max_path": metrics.get("max_path_length", 0),
                    "layer_dist": metrics.get("layer_distribution", {}),
                    "fingerprint_overlap": len(overlap),
                    "overlap_features": [list(x) for x in overlap],
                }

                save_graph(pruned, os.path.join(OUT, f"circuit_{case_id}_{variant}.json"))
                log.info(f"    {label}: nodes={case_results[variant]['n_nodes']}, "
                         f"ff={case_results[variant]['n_ff_edges']}, "
                         f"overlap={len(overlap)}")

            except Exception as e:
                log.error(f"    {label} circuit failed: {e}")
                case_results[variant] = {"error": str(e)}

        results["circuits"][case_id] = case_results

    save_results(results, "exp2_circuits.json")
    return results


# ============================================================
# EXPERIMENT 3: Intervention Validation
# ============================================================

def run_exp3_interventions(model, tokenizer, clt, case_ids, fingerprint_results):
    """Ablate fingerprint features, generate responses, measure behavioral change."""
    log.info("=" * 70)
    log.info("EXPERIMENT 3: INTERVENTION VALIDATION")
    log.info("=" * 70)

    results = {"interventions": {}}

    for case_id in case_ids:
        info = get_case_info(case_id)
        if info["is_multi_turn"]:
            continue

        prompts = get_case_prompts(case_id)
        top_feats = fingerprint_results.get("cases", {}).get(case_id, {}).get("top_features", [])

        if not top_feats or not prompts:
            continue

        log.info(f"\n  {info['name']}: intervening...")

        feature_specs = [(f["layer"], f["feature_idx"]) for f in top_feats[:30]]
        trigger_text = prompts[0]["trigger"]

        case_results = {"feature_specs": [list(s) for s in feature_specs]}

        # Generate with ablation on trigger
        try:
            gen = generate_with_ablation(
                model, clt, tokenizer, trigger_text,
                feature_specs=feature_specs,
                max_new_tokens=120, amplification=3.0)
            metrics = compare_generations(gen)

            case_results["trigger"] = {
                "clean": gen.generation_clean[:500],
                "ablated": gen.generation_intervened[:500],
                "metrics_clean": metrics["clean"],
                "metrics_ablated": metrics["intervened"],
                "deltas": metrics["deltas"],
            }

            log.info(f"    CLEAN: {gen.generation_clean[:100]}...")
            log.info(f"    ABLATED: {gen.generation_intervened[:100]}...")

            # Log significant metric changes
            for key in ["agreement_score", "empathy_score", "refusal_score",
                        "hedging_score", "technical_score"]:
                c = metrics["clean"].get(key, 0)
                a = metrics["intervened"].get(key, 0)
                if c != a:
                    log.info(f"      {key}: {c} → {a}")

        except Exception as e:
            log.error(f"    Trigger intervention failed: {e}")
            log.error(traceback.format_exc())
            case_results["trigger"] = {"error": str(e)}

        # Also run on control (should show less change)
        try:
            control_text = prompts[0]["control"]
            gen_ctrl = generate_with_ablation(
                model, clt, tokenizer, control_text,
                feature_specs=feature_specs,
                max_new_tokens=120, amplification=3.0)
            metrics_ctrl = compare_generations(gen_ctrl)

            case_results["control"] = {
                "clean": gen_ctrl.generation_clean[:500],
                "ablated": gen_ctrl.generation_intervened[:500],
                "metrics_clean": metrics_ctrl["clean"],
                "metrics_ablated": metrics_ctrl["intervened"],
                "deltas": metrics_ctrl["deltas"],
            }

        except Exception as e:
            log.error(f"    Control intervention failed: {e}")
            case_results["control"] = {"error": str(e)}

        results["interventions"][case_id] = case_results

    save_results(results, "exp3_interventions.json")
    return results


# ============================================================
# EXPERIMENT 4: Verdict Classification
# ============================================================

def compute_verdict(case_id, fingerprint, circuit, intervention):
    """Classify behavior based on accumulated evidence."""
    info = get_case_info(case_id)

    # Evidence scoring
    score = 0
    evidence = {}

    # 1. Fingerprint strength
    top_feats = fingerprint.get("top_features", [])
    if top_feats:
        top_strength = top_feats[0]["mean_abs_diff"]
        evidence["fingerprint_strength"] = top_strength
        if top_strength > 200:
            score += 1
        if top_strength > 500:
            score += 1

    # 2. Circuit coherence
    trigger_circuit = circuit.get("trigger", {})
    if "error" not in trigger_circuit:
        ff = trigger_circuit.get("n_ff_edges", 0)
        overlap = trigger_circuit.get("fingerprint_overlap", 0)
        evidence["ff_edges"] = ff
        evidence["fingerprint_in_circuit"] = overlap
        if ff > 10:
            score += 1
        if overlap > 3:
            score += 1

    # 3. Intervention effect
    trigger_int = intervention.get("trigger", {})
    if "error" not in trigger_int:
        deltas = trigger_int.get("deltas", {})
        any_change = any(abs(v) > 0 for k, v in deltas.items()
                        if k.startswith("delta_"))
        evidence["intervention_caused_change"] = any_change
        if any_change:
            score += 1

    # 4. Control vs trigger differential
    control_int = intervention.get("control", {})
    if "error" not in trigger_int and "error" not in control_int:
        t_deltas = trigger_int.get("deltas", {})
        c_deltas = control_int.get("deltas", {})
        t_total = sum(abs(v) for v in t_deltas.values() if isinstance(v, (int, float)))
        c_total = sum(abs(v) for v in c_deltas.values() if isinstance(v, (int, float)))
        evidence["trigger_effect_vs_control"] = t_total / max(c_total, 0.01)
        if t_total > c_total * 1.5:
            score += 1

    # Verdict
    if score >= 4:
        verdict, confidence = "Genuine", "High"
    elif score >= 3:
        verdict, confidence = "Genuine", "Medium"
    elif score >= 2:
        verdict, confidence = "Confused", "Medium"
    elif score >= 1:
        verdict, confidence = "Confused", "Low"
    else:
        verdict, confidence = "Emergent Artifact", "Low"

    return {
        "case_id": case_id,
        "case_name": info["name"],
        "expected_verdict": info["expected_verdict"],
        "actual_verdict": verdict,
        "confidence": confidence,
        "evidence_score": score,
        "max_score": 6,
        "evidence": evidence,
        "match": verdict.lower() in info["expected_verdict"].lower(),
    }


def run_exp4_verdicts(case_ids, fingerprints, circuits, interventions):
    """Generate verdict table from all evidence."""
    log.info("=" * 70)
    log.info("EXPERIMENT 4: VERDICT TABLE")
    log.info("=" * 70)

    results = {"verdicts": []}

    for case_id in case_ids:
        info = get_case_info(case_id)
        if info["is_multi_turn"]:
            continue

        fp = fingerprints.get("cases", {}).get(case_id, {})
        ci = circuits.get("circuits", {}).get(case_id, {})
        iv = interventions.get("interventions", {}).get(case_id, {})

        verdict = compute_verdict(case_id, fp, ci, iv)
        results["verdicts"].append(verdict)

        match_str = "✓" if verdict["match"] else "✗"
        log.info(f"  {verdict['case_name']:<30s} "
                 f"Expected: {verdict['expected_verdict']:<15s} "
                 f"Got: {verdict['actual_verdict']:<15s} "
                 f"[{verdict['evidence_score']}/{verdict['max_score']}] {match_str}")

    # Cross-behavior overlap
    log.info("\n  CROSS-BEHAVIOR FEATURE OVERLAP:")
    case_feature_sets = {}
    for case_id in case_ids:
        if get_case_info(case_id)["is_multi_turn"]:
            continue
        feats = fingerprints.get("cases", {}).get(case_id, {}).get("top_features", [])
        case_feature_sets[case_id] = {(f["layer"], f["feature_idx"]) for f in feats[:20]}

    overlap_matrix = {}
    case_list = list(case_feature_sets.keys())
    for i, c1 in enumerate(case_list):
        for j, c2 in enumerate(case_list):
            if i < j:
                intersection = len(case_feature_sets[c1] & case_feature_sets[c2])
                union = len(case_feature_sets[c1] | case_feature_sets[c2])
                jaccard = intersection / max(union, 1)
                overlap_matrix[f"{c1}_vs_{c2}"] = {
                    "jaccard": jaccard, "shared": intersection, "union": union,
                }
                if jaccard > 0.05:
                    n1 = get_case_info(c1)["name"]
                    n2 = get_case_info(c2)["name"]
                    log.info(f"    {n1} × {n2}: Jaccard={jaccard:.3f}, shared={intersection}")

    results["cross_behavior_overlap"] = overlap_matrix

    save_results(results, "exp4_verdicts.json")
    return results


# ============================================================
# SPECIAL: C11 Multi-Turn Escalation
# ============================================================

def run_multiturn_c11(model, tokenizer, clt):
    """Track sycophancy escalation across 5 turns."""
    log.info("=" * 70)
    log.info("SPECIAL: C11 MULTI-TURN SYCOPHANCY ESCALATION")
    log.info("=" * 70)

    case = CASES.get("C11_sycophancy_escalation")
    if not case:
        log.warning("C11 not found")
        return {}

    results = {"escalation": []}

    for prompt_data in case["prompts"]:
        turns = prompt_data["turns"]
        control = prompt_data["control"]

        log.info(f"\n  {prompt_data['notes']}")
        log.info(f"  {len(turns)} escalation turns")

        # Control baseline
        try:
            inp_c = tokenizer.encode(control, return_tensors="pt",
                                      add_special_tokens=True).to("cuda")
            clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inp_c)
            if next(clt.parameters()).dtype == torch.float16:
                clt_in = clt_in.half()
            feats_control = clt.encode(clt_in)[-1].float().cpu()
        except Exception as e:
            log.error(f"  Control failed: {e}")
            continue

        turn_results = []
        for ti, turn_text in enumerate(turns):
            try:
                inp_t = tokenizer.encode(turn_text, return_tensors="pt",
                                          add_special_tokens=True).to("cuda")
                clt_in, _ = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inp_t)
                if next(clt.parameters()).dtype == torch.float16:
                    clt_in = clt_in.half()
                feats_turn = clt.encode(clt_in)[-1].float().cpu()

                delta = feats_turn - feats_control
                flat = delta.abs().flatten()
                top_vals, top_flat = flat.topk(10)
                d_sae = delta.shape[1]

                top_features = []
                for fi, val in zip(top_flat, top_vals):
                    layer = fi.item() // d_sae
                    feat = fi.item() % d_sae
                    top_features.append({
                        "layer": layer, "feature": feat,
                        "delta": delta[layer, feat].item(),
                    })

                # Generate response for this turn
                with torch.no_grad():
                    gen_out = model.generate(input_ids=inp_t, max_new_tokens=80, do_sample=False)
                gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=False)
                if "<start_of_turn>model" in gen_text:
                    gen_text = gen_text.split("<start_of_turn>model")[-1].split("<end_of_turn>")[0].strip()

                beh_metrics = compute_behavioral_metrics(gen_text)

                turn_results.append({
                    "turn": ti + 1,
                    "top_features": top_features,
                    "total_diff_magnitude": float(delta.abs().sum()),
                    "generation": gen_text[:300],
                    "behavioral_metrics": beh_metrics,
                })

                log.info(f"    Turn {ti+1}: agree={beh_metrics.get('agreement_score', 0)}, "
                         f"hedge={beh_metrics.get('hedging_score', 0)}, "
                         f"|Δ|={delta.abs().sum():.0f}")

            except Exception as e:
                log.error(f"    Turn {ti+1} failed: {e}")

        results["escalation"].append({
            "notes": prompt_data["notes"],
            "n_turns": len(turns),
            "turns": turn_results,
        })

    save_results(results, "exp_c11_escalation.json")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MI Detective — Complete Experiment Suite")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", type=str, default=None, help="Run specific case (e.g., C2)")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated cases (e.g., C1,C2,C9)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["fingerprint", "circuits", "interventions", "verdicts", "multiturn"])
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("MI DETECTIVE — COMPLETE EXPERIMENT SUITE")
    log.info(f"  Time: {datetime.now().isoformat()}")
    log.info("=" * 70)

    model, tokenizer = load_gemma3_1b("it", device="cuda")

    log.info("Loading CLT-IT (affine)...")
    clt = load_clt(width="262k", l0="big", affine=True, variant="it",
                   device="cuda", half_precision=True, cache_dir=CACHE)
    log.info(f"GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    # Select cases
    all_cases = get_all_case_ids()
    if args.case:
        # Match partial: --case C2 matches C2_jailbreak_patterns
        case_ids = [c for c in all_cases if c.startswith(args.case)]
        if not case_ids:
            case_ids = [args.case]
    elif args.cases:
        requested = args.cases.split(",")
        case_ids = []
        for r in requested:
            matches = [c for c in all_cases if c.startswith(r.strip())]
            case_ids.extend(matches)
    elif args.quick:
        case_ids = ["C1_sycophantic_validation", "C2_jailbreak_patterns",
                    "C9_emotional_dependency"]
    else:
        case_ids = all_cases

    # Remove multi-turn from standard pipeline (handled separately)
    standard_cases = [c for c in case_ids if not get_case_info(c).get("is_multi_turn", False)]
    has_multiturn = any(get_case_info(c).get("is_multi_turn", False) for c in case_ids)

    log.info(f"Cases: {case_ids}")
    log.info(f"  Standard: {len(standard_cases)}, Multi-turn: {has_multiturn}")

    t_start = time.time()

    # Exp 1
    fingerprints = None
    if args.only is None or args.only == "fingerprint":
        fingerprints = run_exp1_fingerprint(model, tokenizer, clt, standard_cases)
    else:
        fp_path = os.path.join(OUT, "exp1_fingerprints.json")
        if os.path.exists(fp_path):
            with open(fp_path) as f:
                fingerprints = json.load(f)

    if fingerprints is None:
        log.error("No fingerprint results. Run fingerprinting first.")
        return

    # Exp 2
    circuits = {"circuits": {}}
    if args.only is None or args.only == "circuits":
        circuits = run_exp2_circuits(model, tokenizer, clt, standard_cases, fingerprints)
    else:
        ci_path = os.path.join(OUT, "exp2_circuits.json")
        if os.path.exists(ci_path):
            with open(ci_path) as f:
                circuits = json.load(f)

    # Exp 3
    interventions = {"interventions": {}}
    if args.only is None or args.only == "interventions":
        interventions = run_exp3_interventions(model, tokenizer, clt, standard_cases, fingerprints)
    else:
        iv_path = os.path.join(OUT, "exp3_interventions.json")
        if os.path.exists(iv_path):
            with open(iv_path) as f:
                interventions = json.load(f)

    # Exp 4
    if args.only is None or args.only == "verdicts":
        run_exp4_verdicts(standard_cases, fingerprints, circuits, interventions)

    # Multi-turn
    if has_multiturn and (args.only is None or args.only == "multiturn"):
        run_multiturn_c11(model, tokenizer, clt)

    total = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info(f"MI DETECTIVE COMPLETE — {total/60:.1f} min")
    log.info("=" * 70)
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            log.info(f"  {f}: {os.path.getsize(os.path.join(OUT, f))/1024:.0f} KB")


if __name__ == "__main__":
    main()

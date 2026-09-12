"""Hypo V12.2: conservative temporal safety gate.

Keeps V12.1's temporal protocol unchanged:
- Days 1-21: personal fine-tuning.
- Days 22-30: untouched temporal safety gate.
- Day 31+: blind future test.

Only the activation rule changes. A personalized candidate must demonstrate a
material recall improvement (>= 5 percentage points) on the gate window. Equal
recall plus fewer notifications is no longer sufficient to activate it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lopo_hypo_v12_1 as base

MIN_RECALL_GAIN_PP = 5.0
MIN_WARNING_MINUTES = 15.0
MAX_NOTIFICATION_INCREASE = 0.50


def conservative_gate_decision(pop_ev, pop_nt, cand_ev, cand_nt):
    """Activate personalization only on demonstrated material recall benefit."""
    pop_recall = float(pop_ev["event_recall"])
    cand_recall = float(cand_ev["event_recall"])
    gain_pp = (cand_recall - pop_recall) * 100.0
    warning = cand_ev["median_warning_minutes"]

    if warning is None or float(warning) < MIN_WARNING_MINUTES:
        return False, "blocked_warning_time", gain_pp

    if gain_pp < MIN_RECALL_GAIN_PP:
        return False, "blocked_insufficient_recall_gain", gain_pp

    if cand_nt is None or pop_nt is None:
        return False, "blocked_missing_notification_metric", gain_pp

    if float(cand_nt) > float(pop_nt) + MAX_NOTIFICATION_INCREASE:
        return False, "blocked_excess_notifications_despite_recall_gain", gain_pp

    return True, "activated_material_recall_gain", gain_pp


def requested_output_dir(argv):
    if "--output-dir" in argv:
        i = argv.index("--output-dir")
        if i + 1 < len(argv):
            return Path(argv[i + 1])
    return Path("models/v12_2_conservative_safety_gate")


def postprocess(outdir: Path):
    old_csv = outdir / "v12_1_per_patient.csv"
    new_csv = outdir / "v12_2_per_patient.csv"
    if old_csv.exists():
        old_csv.replace(new_csv)

    old_report = outdir / "v12_1_report.json"
    new_report = outdir / "v12_2_report.json"
    if old_report.exists():
        report = json.loads(old_report.read_text())
        report["model"] = "hypo_v12_2_conservative_temporal_safety_gate"
        report["experiment_change_from_v12_1"] = (
            "Activation requires >=5 pp gate recall gain; equal recall plus lower "
            "notification burden is no longer sufficient."
        )
        report["gate_rules"] = {
            "minimum_finetune_events": base.MIN_FT_EVENTS,
            "minimum_gate_events": base.MIN_GATE_EVENTS,
            "minimum_recall_gain_pp": MIN_RECALL_GAIN_PP,
            "minimum_warning_minutes": MIN_WARNING_MINUTES,
            "max_notification_increase_per_day": MAX_NOTIFICATION_INCREASE,
            "equal_recall_lower_notifications_can_activate": False,
        }
        new_report.write_text(json.dumps(report, indent=2))
        old_report.unlink()


if __name__ == "__main__":
    # run_patient() in V12.1 resolves gate_decision from its module globals, so
    # replacing it here changes exactly one experimental variable.
    base.gate_decision = conservative_gate_decision
    outdir = requested_output_dir(sys.argv)
    print(
        "V12.2 conservative gate: personalization requires >=5 pp recall gain "
        "on days 22-30.",
        flush=True,
    )
    base.main()
    postprocess(outdir)
    print(f"V12.2 completed successfully. Results: {outdir}", flush=True)

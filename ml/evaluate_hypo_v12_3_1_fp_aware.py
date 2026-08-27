"""Offline research evaluation for V12.3.1.

Retrospective benchmark only; not for clinical use or live alerting.
Keeps the V12.2 temporal split and conservative personalization gate, while
comparing candidate alert policies on days 22-30. An alternative policy is
selected only when it preserves gate recall and warning time, does not increase
notification burden, and reduces false-alert episodes. Day 31+ remains blind.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import lopo_hypo_v12_1 as base
from lopo_hypo_v12_2 import conservative_gate_decision

MIN_WARNING_MINUTES=15.0
EPS=1e-12
_reference_choose=base.choose

def _params(row):
 return {"threshold":float(row.threshold),"persistence":int(row.persistence),"clear_steps":int(row.clear_steps),"rearm_margin":float(row.rearm_margin)}

def fp_aware_choose(g):
 reference,tag=_reference_choose(g)
 mask=np.ones(len(g),dtype=bool)
 for k,v in reference.items(): mask &= np.isclose(g[k].astype(float),float(v))
 ref=g.loc[mask].iloc[0]
 candidates=g[(g.recall>=float(ref.recall)-EPS)&(g.warning>=MIN_WARNING_MINUTES)&(g.notifications<=float(ref.notifications)+EPS)&(g.false_alerts<float(ref.false_alerts)-EPS)]
 if candidates.empty: return reference,"fp_aware_keep_reference"
 row=candidates.sort_values(["false_alerts","recall","notifications","warning"],ascending=[True,False,True,False]).iloc[0]
 return _params(row),"fp_aware_lower_false_alerts_noninferior_recall"

def outdir(argv):
 if "--output-dir" in argv:
  i=argv.index("--output-dir")
  if i+1<len(argv): return Path(argv[i+1])
 return Path("ml/models/v12_3_1_fp_aware")

def postprocess(d):
 old=d/"v12_1_per_patient.csv"; new=d/"v12_3_1_per_patient.csv"
 if old.exists(): old.replace(new)
 oldr=d/"v12_1_report.json"; newr=d/"v12_3_1_report.json"
 if oldr.exists():
  r=json.loads(oldr.read_text()); r["model"]="hypo_v12_3_1_fp_aware_offline_evaluation"; r["baseline"]="V12.2"
  r["experiment_change_from_v12_2"]="Offline gate-window policy comparison prioritizes fewer false-alert episodes only when recall is non-inferior, warning >=15 min, and notifications do not increase. Day 31+ remains blind."
  r["clinical_status"]="retrospective research evaluation only; not clinically validated; not for live alerting"
  newr.write_text(json.dumps(r,indent=2)); oldr.unlink()

if __name__=="__main__":
 base.choose=fp_aware_choose
 base.gate_decision=conservative_gate_decision
 d=outdir(sys.argv)
 print("V12.3.1 offline FP-aware evaluation; V12.2 gate preserved.",flush=True)
 base.main(); postprocess(d)
 print(f"V12.3.1 completed. Results: {d}",flush=True)

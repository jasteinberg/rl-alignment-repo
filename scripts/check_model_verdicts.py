"""
Precheck: can pythia-2.8b even answer TRUE/FALSE on these statements?

Steering is only meaningful if the model has a verdict to flip. We measure
logit(" TRUE") - logit(" FALSE") at the final position of a prompt, and check
that it correlates with the ground-truth label. If AUROC ~ 0.5 the model has no
behavioural signal and a steering experiment measures nothing.

Drafted with the assistance of Claude (Anthropic).
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("s", os.path.join(REPO, "scripts/snr_sweep.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

from utils.env import ENV

D = ENV.GOT
dev = "mps"

TEMPLATE = '{stmt} This statement is:'

def verdict_logits(model, tok, stmts, tid_true, tid_false, bs=8):
    out = []
    with torch.no_grad():
        for i in range(0, len(stmts), bs):
            prompts = [TEMPLATE.format(stmt=s) for s in stmts[i:i+bs]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            b = torch.arange(last.shape[0], device=dev)
            fin = logits[b, last].float()
            out.append((fin[:, tid_true] - fin[:, tid_false]).cpu().numpy())
    return np.concatenate(out)

for mname in ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]:
    tok, model = S.get_model(mname, dev, torch.float16)
    tt = tok(" TRUE").input_ids[0]; tf = tok(" FALSE").input_ids[0]
    print(f"\n=== {mname}  (' TRUE'={tt}, ' FALSE'={tf}) ===")
    for ds in ["cities", "companies_true_false", "counterfact_true_false"]:
        df = pd.read_csv(f"{D}/{ds}.csv").dropna(subset=["statement","label"])
        df["label"] = df["label"].astype(int)
        df = df.sample(n=min(200, len(df)), random_state=0)
        z = verdict_logits(model, tok, df["statement"].tolist(), tt, tf)
        y = df["label"].to_numpy()
        a = roc_auc_score(y, z)
        acc = ((z > np.median(z)).astype(int) == y).mean()
        print(f"  {ds:<26} behavioural AUROC={a:.3f}  (acc@median={acc:.3f})  "
              f"mean(logit diff)={z.mean():+.2f}")
    del model, tok
    import gc; gc.collect(); torch.mps.empty_cache()

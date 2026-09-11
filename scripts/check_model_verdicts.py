"""
Can pythia-2.8b judge these statements true or false on its own, before any steering?

This is why the steering harness scores factual-preference completions rather than
a judgment: the judgment readout is a behavior a base model barely has, so steering
it would measure nothing.

  --mode judgment (default)  the number the post quotes: "{stmt} This statement is:"
      scored as log P(" true") - log P(" false") at the final token, AUROC against the
      labels, pythia-2.8b, class-balanced cap 1199 at seed 0. Writes
      artifacts/check_model_verdicts.json.
  --mode precheck            the original July precheck: logit(" TRUE") - logit(" FALSE")
      on 200 sampled statements, three Pythia sizes, printed only.

Drafted with the assistance of Claude (Anthropic).
"""
import argparse, gc, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from truthlib import acts, data

from utils.env import ENV

D = ENV.GOT
dev = "mps"
OUT = os.path.join(REPO, "artifacts", "check_model_verdicts.json")

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


def judgment_scores(model, tok, stmts, bs=16):
    """log P(" true") - log P(" false") after the judgment prompt, final real token."""
    ids_t = tok(" true", add_special_tokens=False).input_ids
    ids_f = tok(" false", add_special_tokens=False).input_ids
    assert len(ids_t) == 1 and len(ids_f) == 1, (ids_t, ids_f)
    out = []
    for i in range(0, len(stmts), bs):
        b = [TEMPLATE.format(stmt=s) for s in stmts[i:i + bs]]
        enc = tok(b, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            lg = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        lp = lg[torch.arange(len(b)), last].float().log_softmax(-1)
        out += (lp[:, ids_t[0]] - lp[:, ids_f[0]]).cpu().tolist()
    return np.array(out)


def judgment(out_path):
    mname = "EleutherAI/pythia-2.8b"
    tok, model = acts.get_model(mname, dev, torch.float16)
    res = {}
    for ds in ["cities", "common_claim_true_false", "counterfact_true_false"]:
        st, y = data.load_dataset(ds, cap=1199, seed=0)
        sc = judgment_scores(model, tok, st)
        res[ds] = {"n": int(len(y)), "auroc": float(roc_auc_score(y, sc)),
                   "mean_score_true": float(sc[y == 1].mean()),
                   "mean_score_false": float(sc[y == 0].mean())}
        print(f"  {ds:<26} n={len(y)}  judgment AUROC={res[ds]['auroc']:.3f}", flush=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"model": mname, "template": TEMPLATE, "targets": [" true", " false"],
                              "cap": 1199, "seed": 0, "dtype": "float16", "device": dev},
                   "results": res}, f, indent=1)
    print(f"wrote {out_path}")


def precheck():
    for mname in ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]:
        tok, model = acts.get_model(mname, dev, torch.float16)
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
        gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["judgment", "precheck"], default="judgment")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    judgment(args.out) if args.mode == "judgment" else precheck()

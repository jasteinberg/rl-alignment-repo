#!/bin/zsh
# Null power extension (Julia's spec): 400 at cf deep + cities L28, 100 elsewhere.
# ALWAYS pass --alphas explicitly. Scale bug fixed in c707713 (class_gap, was std(X@th)).
cd "$(dirname "$0")/.."
# Interpreter: override with PY=... (defaults to whatever python3 is on PATH).
PY=${PY:-python3}
$PY -W ignore scripts/extend_null.py --models EleutherAI/pythia-2.8b --datasets counterfact_true_false --layers 20,24,28 --alphas 0.5,1,2,4 --n_rand 400
$PY -W ignore scripts/extend_null.py --models EleutherAI/pythia-2.8b --datasets cities --layers 28 --alphas 0.5,1,2,4 --n_rand 400
$PY -W ignore scripts/extend_null.py --models EleutherAI/pythia-2.8b --datasets counterfact_true_false --layers 8,12,16 --alphas 0.5,1,2,4 --n_rand 100
$PY -W ignore scripts/extend_null.py --models EleutherAI/pythia-2.8b --datasets cities --layers 8,12,16,20,24 --alphas 0.5,1,2,4 --n_rand 100
$PY -W ignore scripts/extend_null.py --models EleutherAI/pythia-1.4b --datasets cities,counterfact_true_false --layers 6,9,12,15,18,21 --alphas 0.5,1,2,4 --n_rand 100
echo "null extension chain done $(date)"

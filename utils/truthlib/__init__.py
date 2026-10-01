"""Shared library for the truth-directions scripts.

    estimators  d', AUROC, mass-mean and Fisher directions, within-class spectra,
                massive coordinates, the gradient-overlap bootstrap
    nulls       random-direction null, shuffled-label (Cover) sweep, the
                per-dataset excess curve, its power-law fit and PR collapse
    sweep       the per-layer fit / orient / held-out-score protocol, layer
                selection, the superposition probe, transfer fits
    geometry    observables at every layer, massive coordinates, the rogue
                dimension and its steering arms, the outlier appendix
    shrinkage   Ledoit-Wolf intensity and its decomposition, the shrinkage
                path, cross-validated rho
                (estimators through shrinkage: NumPy, SciPy, scikit-learn only)
    data        Marks & Tegmark datasets and contrastive completion pairs
    acts        model loading, activation extraction, the unsteered verdict readout (torch)
    steering    residual-stream hook, behavioral score, steering cells and nulls,
                checkpoints and the chi summary, per-pair score gradients (torch)
    refusal     refusal-direction prompts, log-odds score, gradients, Qwen hook (torch)

Drafted with the assistance of Claude (Anthropic).
"""

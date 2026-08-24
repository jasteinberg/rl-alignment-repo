"""Schematic: the behavioral score ell(x) and the odd/even split of the steering
response.

Left panel: one prompt, two completions, ell as the log-probability difference.
Right panel: the +h and -h steered passes and the antisymmetric/symmetric parts
A and S of Delta(h).

Pure schematic -- no data dependencies. Example pair is the counterfact
'.NET Framework' pair used in the post's text.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_behavioral_score.png"

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 150,
    "mathtext.fontset": "dejavuserif",
})
C_TRUE = "#2c6fbb"
C_FALSE = "#c0392b"
C_BOX = "#f2f0ea"
C_EDGE = "#8a8578"
C_STEER = "#7a5aa0"


def box(ax, xy, w, h, text, fc=C_BOX, ec=C_EDGE, fs=10, mono=False, tc="black"):
    bx = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                        fc=fc, ec=ec, lw=0.9)
    ax.add_patch(bx)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc,
            family="monospace" if mono else "serif")
    return xy[0] + w / 2, xy[1] + h / 2


def arrow(ax, p0, p1, color=C_EDGE, lw=1.2, style="-|>", shrink=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                                 shrinkA=shrink, shrinkB=shrink,
                                 mutation_scale=12))


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                               gridspec_kw={"width_ratios": [1.05, 1.0]})
for ax in (axL, axR):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ---------------- left: the score --------------------------------------------
axL.set_title(r"The behavioral score $\ell(x)$", fontsize=12, pad=8)

pc = box(axL, (0.06, 0.42), 0.42, 0.16,
         '".NET Framework is\ncreated by"', mono=True, fs=9)
axL.text(0.27, 0.63, "prompt", ha="center", fontsize=9, color=C_EDGE)

tc_ = box(axL, (0.62, 0.66), 0.30, 0.13, '" Microsoft"', mono=True, fs=9,
          ec=C_TRUE, tc=C_TRUE)
fc_ = box(axL, (0.62, 0.21), 0.30, 0.13, '" Google"', mono=True, fs=9,
          ec=C_FALSE, tc=C_FALSE)
axL.text(0.77, 0.82, "true completion", ha="center", fontsize=9, color=C_TRUE)
axL.text(0.77, 0.145, "false completion", ha="center", fontsize=9, color=C_FALSE)

arrow(axL, (0.48, 0.54), (0.62, 0.72), color=C_TRUE)
arrow(axL, (0.48, 0.46), (0.62, 0.28), color=C_FALSE)

axL.text(0.55, 0.665, r"$\log P_{\mathrm{true}}$", fontsize=10, color=C_TRUE,
         ha="center", rotation=22)
axL.text(0.55, 0.32, r"$\log P_{\mathrm{false}}$", fontsize=10, color=C_FALSE,
         ha="center", rotation=-22)

axL.text(0.5, 0.02,
         r"$\ell(x) = \log P_{\mathrm{true}} - \log P_{\mathrm{false}}$"
         "\n(summed over completion tokens; same prompt, so its likelihood cancels)",
         ha="center", va="bottom", fontsize=10)

# ---------------- right: the odd/even split ----------------------------------
axR.set_title(r"Steering response: odd and even parts", fontsize=12, pad=8)

x0 = box(axR, (0.05, 0.42), 0.26, 0.16, "$x$\nclean pass", fs=10)
xp = box(axR, (0.55, 0.68), 0.34, 0.14, r"$x + h\,c\,w$", fs=10, ec=C_STEER)
xm = box(axR, (0.55, 0.28), 0.34, 0.14, r"$x - h\,c\,w$", fs=10, ec=C_STEER)

arrow(axR, (0.31, 0.54), (0.55, 0.75), color=C_STEER)
arrow(axR, (0.31, 0.46), (0.55, 0.35), color=C_STEER)
axR.text(0.42, 0.70, r"$+h$", fontsize=10, color=C_STEER)
axR.text(0.42, 0.33, r"$-h$", fontsize=10, color=C_STEER)

axR.text(0.72, 0.86, r"$\Delta(+h)=\langle\ell\rangle_{+h}-\langle\ell\rangle_0$",
         ha="center", fontsize=10)
axR.text(0.72, 0.21, r"$\Delta(-h)=\langle\ell\rangle_{-h}-\langle\ell\rangle_0$",
         ha="center", fontsize=10)

axR.text(0.5, 0.02,
         r"$A=\frac{1}{2}\left[\Delta(+h)-\Delta(-h)\right]$: odd, a signed direction"
         "\n"
         r"$S=\frac{1}{2}\left[\Delta(+h)+\Delta(-h)\right]$: even, generic disruption",
         ha="center", va="bottom", fontsize=10)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

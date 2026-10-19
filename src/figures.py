"""Static figures for the README (reports/figures/*.png).   python -m src.figures"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES, PROCESSED, REPORTS, TEST_QUANTILE, VAL_QUANTILE

INK, SOFT, RULE = "#1C2333", "#5A6478", "#E3E6EC"
MODEL_COLOR = {"popularity": "#8A94A6", "demographic_popularity": "#B07D00", "content": "#D4562E",
               "item_knn": "#2A6FD6", "als": "#6250D6", "hybrid": "#13976A", "hybrid_onboard": "#0B5E42"}
MODEL_NAME = {"popularity": "Popularity", "demographic_popularity": "Popular in age/gender group",
              "content": "Content (genres)", "item_knn": "Item-kNN", "als": "ALS (matrix fact.)",
              "hybrid": "Hybrid", "hybrid_onboard": "Hybrid + demographics"}
SEGMENT_NAME = {"warm": "Returning users (1,122)", "new_cold": "New users, nothing rated (640)",
                "new_onboard": "New users after 3 sign-up picks (640)"}

plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": RULE, "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.titlecolor": INK,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def ndcg_by_segment():
    m = pd.read_csv(REPORTS / "metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [6, 3, 7]})
    for ax, seg in zip(axes, ["warm", "new_cold", "new_onboard"]):
        d = m[m["segment"] == seg].sort_values("ndcg")
        y = np.arange(len(d))
        ax.barh(y, d["ndcg"], color=[MODEL_COLOR[x] for x in d["model"]], height=0.62)
        ax.errorbar(d["ndcg"], y, xerr=[d["ndcg"] - d["ndcg_lo"], d["ndcg_hi"] - d["ndcg"]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        ax.set_yticks(y, [MODEL_NAME[x] for x in d["model"]], fontsize=9)
        for yi, v, hi in zip(y, d["ndcg"], d["ndcg_hi"]):
            ax.text(hi + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 0.5)
        ax.set_title(SEGMENT_NAME[seg], fontsize=10)
        ax.set_xlabel("NDCG@10 (95% CI)")
        ax.grid(axis="x", color=RULE, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "ndcg_by_segment.png", bbox_inches="tight")


def ndcg_by_history():
    d = pd.read_csv(REPORTS / "by_history.csv")
    order = ["0-9", "10-29", "30-99", "100+"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in ["popularity", "item_knn", "als", "hybrid"]:
        g = d[d["model"] == m].set_index("liked_before_cutoff").reindex(order)
        ax.plot(order, g["mean"], marker="o", color=MODEL_COLOR[m], label=MODEL_NAME[m], linewidth=2)
    counts = d[d["model"] == "hybrid"].set_index("liked_before_cutoff").reindex(order)["count"]
    ax.set_xticks(range(len(order)), [f"{o}\n(n={int(c)})" for o, c in zip(order, counts)])
    ax.set_xlabel("Movies the user liked before the cutoff")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Personalisation pays off once a user has history", fontsize=11)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "ndcg_by_history.png", bbox_inches="tight")


def timeline():
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    t_val, t_test = r["ts"].quantile(VAL_QUANTILE), r["ts"].quantile(TEST_QUANTILE)
    week = pd.to_datetime(r["ts"], unit="s").dt.to_period("W").dt.start_time
    counts = week.groupby([week, r["split"]]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    colors = {"train": "#2A6FD6", "val": "#B07D00", "test": "#13976A"}
    for s in ["train", "val", "test"]:
        ax.bar(counts.index, counts[s], width=7, color=colors[s], label=s, linewidth=0)
    for t in [t_val, t_test]:
        ax.axvline(pd.to_datetime(t, unit="s"), color=INK, linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("Ratings per week (log)")
    ax.set_title("Global time split: the model only sees ratings made before each cutoff", fontsize=11)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "timeline_split.png", bbox_inches="tight")


def tuning_cf():
    d = pd.read_csv(REPORTS / "tuning_cf.csv")
    best = d.loc[d.groupby(["model", "data"])["ndcg"].idxmax()]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    labels, vals, cols = [], [], []
    for _, row in best.sort_values("ndcg").iterrows():
        labels.append(f"{MODEL_NAME[row['model']]}" + ("" if row["data"] == "-" else f", {row['data']}"))
        vals.append(row["ndcg"])
        cols.append(MODEL_COLOR[row["model"]] if row["data"] != "all ratings" else "#C9CED8")
    ax.barh(labels, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Validation NDCG@10 (best setting each)")
    ax.set_title("Learning only from liked movies is what beats popularity", fontsize=11)
    ax.set_xlim(0, 0.24)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_liked_vs_all.png", bbox_inches="tight")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    ndcg_by_segment(); ndcg_by_history(); timeline(); tuning_cf()
    print("figures written to", FIGURES)

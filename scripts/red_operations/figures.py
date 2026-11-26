"""Figures for docs/red/operational-decisions.md.

usage: python figures.py <analysis_out_dir> <lg5.parquet> <fig_dir>
Palette and mark specs follow the dataviz reference instance (light surface).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activity as A  # noqa: E402
import features as F  # noqa: E402
import planner as P  # noqa: E402
from prices import hourly_prices  # noqa: E402

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# One colour per season in every figure (colour follows the entity, never its rank).
# 2022/23 is a partial season (from Feb 2023), drawn in the neutral grey.
SEASON = {
    "2022/23": "#9c9c9c",
    "2023/24": SERIES[0],
    "2024/25": SERIES[1],
    "2025/26": SERIES[2],
    "2026/27": SERIES[3],
}


def season_colors(seasons):
    return {s: SEASON[s] for s in seasons}


def bar(ax, x, y, color, labels=None, fmt="{:.0f}"):
    bars = ax.bar(x, y, width=0.55, color=color, zorder=3)
    for b, v in zip(bars, y, strict=False):
        if np.isfinite(v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                fmt.format(v),
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=9,
            )
    ax.grid(axis="x", visible=False)
    return bars


def fig_seasons(out: Path, figdir: Path, full: list[str]) -> None:
    k = pd.read_csv(out / "kpis_season.csv", index_col=0).loc[full]
    panels = [
        ("lamp_kwh", "Lamps, kWh/m² (est.)", "{:.0f}", 1),
        ("heat_gj", "Pipe heat, GJ/m² (est.)", "{:.2f}", 1),
        ("unasked_share", "Heat not asked for, %", "{:.0f}", 100),
        ("drain_share", "Drain, % of irrigation", "{:.0f}", 100),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), gridspec_kw={"wspace": 0.35})
    for ax, (col, title, fmt, mul) in zip(axes, panels, strict=False):
        bar(ax, k.index, k[col] * mul, SERIES[0], fmt=fmt)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, (k[col] * mul).max() * 1.2)
    fig.savefig(figdir / "seasons.png")
    plt.close(fig)


def fig_normalised(out: Path, figdir: Path) -> None:
    j = json.loads((out / "normalised.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=False)
    for ax, (key, title) in zip(
        axes,
        [
            ("weather_only", "Daily heat vs 2023/24, at the same weather"),
            ("equal_dT", "…and at the same inside–outside difference"),
        ],
        strict=False,
    ):
        m = j["heat"][key]
        seasons = [m["base"]] + sorted(m["effects"])
        est = [0.0] + [m["effects"][s][0] for s in seasons[1:]]
        se = [0.0] + [m["effects"][s][1] for s in seasons[1:]]
        pct = [e / m["mean_heat"] * 100 for e in est]
        ax.axhline(0, color=TEXT2, linewidth=1)
        ax.errorbar(
            seasons,
            pct,
            yerr=[1.96 * s / m["mean_heat"] * 100 for s in se],
            fmt="o",
            color=SERIES[0],
            ecolor=SERIES[0],
            elinewidth=2,
            capsize=0,
            markersize=7,
            zorder=3,
        )
        for s, p in zip(seasons, pct, strict=False):
            ax.text(s, p, f"  {p:+.0f}%", va="center", color=TEXT, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("% of mean daily heat (Oct–Apr)")
        ax.grid(axis="x", visible=False)
    fig.savefig(figdir / "heat_normalised.png")
    plt.close(fig)


def fig_monthly_lamps(out: Path, figdir: Path) -> None:
    d = pd.read_csv(out / "daily.csv", index_col=0, parse_dates=True)
    d = d[(d.coverage > 0.9)]
    order = [9, 10, 11, 12, 1, 2, 3, 4]
    m = (
        d[d.month.isin(order)]
        .groupby(["season", "month"])[["lamp_h", "light_jcm2", "t24", "heat_mj"]]
        .mean()
    )
    cols = season_colors(m.index.get_level_values(0).unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, (col, title) in zip(
        axes,
        [
            ("lamp_h", "Lamp-hours per day"),
            ("light_jcm2", "Light sum, sun + lamps (J/cm²/day)"),
            ("heat_mj", "Pipe heat, MJ/m²/day (estimated)"),
        ],
        strict=False,
    ):
        for s, g in m.groupby(level=0):
            g = g.droplevel(0).reindex(order)
            ax.plot(range(len(order)), g[col], color=cols[s], marker="o", markersize=4, label=s)
        if col == "light_jcm2":
            ax.axhline(1000, color=TEXT2, linewidth=1)
            ax.text(
                len(order) - 1, 1000, "1000 rule", va="bottom", ha="right", color=TEXT2, fontsize=8
            )
        ax.set_xticks(range(len(order)), ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"])
        ax.set_title(title, fontsize=10)
    axes[0].legend(loc="lower center", ncols=2, fontsize=8)
    fig.savefig(figdir / "monthly.png")
    plt.close(fig)


def fig_planner(planner_csv: Path, figdir: Path) -> None:
    p = pd.read_csv(planner_csv, index_col=0, parse_dates=True)
    p = p.dropna(subset=["cost_same", "cost_t1000_fc"])
    y = np.where(p.index.month >= 7, p.index.year, p.index.year - 1)
    p["season"] = [f"{a}/{str(a + 1)[2:]}" for a in y]
    g = p.groupby("season")[
        ["cost_actual", "cost_same", "cost_t1000_fc", "lamp_h_actual", "lamp_h_t1000_fc"]
    ].sum()
    g = g[p.groupby("season").size() > 60]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    x = np.arange(len(g))
    wdt = 0.26
    names = [
        ("cost_actual", "as run"),
        ("cost_same", "same light, cheapest hours"),
        ("cost_t1000_fc", "1000 J/cm², re-planned hourly"),
    ]
    for i, (c, lab) in enumerate(names):
        bars = axes[0].bar(
            x + (i - 1) * wdt, g[c], width=wdt - 0.03, color=SERIES[i], label=lab, zorder=3
        )
        for b, v in zip(bars, g[c], strict=False):
            axes[0].text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT,
            )
    axes[0].set_xticks(x, g.index)
    axes[0].set_title("Lamp electricity at day-ahead price, €/m² per lamp season", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="x", visible=False)
    axes[0].set_ylim(0, g[[c for c, _ in names]].max().max() * 1.35)
    for i, (c, lab) in enumerate(
        [("lamp_h_actual", "as run"), ("lamp_h_t1000_fc", "1000 J/cm², re-planned hourly")]
    ):
        bars = axes[1].bar(
            x + (i - 0.5) * wdt,
            g[c],
            width=wdt - 0.03,
            color=SERIES[0 if i == 0 else 2],
            label=lab,
            zorder=3,
        )
        for b, v in zip(bars, g[c], strict=False):
            axes[1].text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT,
            )
    axes[1].set_xticks(x, g.index)
    axes[1].set_title("Lamp-hours per lamp season", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="x", visible=False)
    axes[1].set_ylim(0, g[["lamp_h_actual", "lamp_h_t1000_fc"]].max().max() * 1.35)
    fig.savefig(figdir / "planner.png")
    plt.close(fig)


def fig_activity(f: pd.DataFrame, figdir: Path) -> None:
    q = A.quiet_rate(f)
    ev_on = A.lamp_on_events(f)
    ev_sr = A.sunrise_events(f)
    ev_sr = ev_sr[np.isin(ev_sr.month, [4, 5, 6, 7, 8])]
    on = f["lamp_frac"] > 0.5
    off = (~on) & on.shift(1, fill_value=False)
    ev_off = f.index[off & (f["sun"] <= 0)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, (ev, title, before, after) in zip(
        axes,
        [
            (ev_on, f"Lamps on, in the dark (n={len(ev_on)})", 2, 4),
            (ev_off, f"Lamps off, in the dark (n={len(ev_off)})", 2, 4),
            (ev_sr, f"Sunrise, no lamps, Apr–Aug (n={len(ev_sr)})", 2, 2),
        ],
        strict=False,
    ):
        al = A.aligned(q.E, ev, before, after) * 1000
        n = al.notna().sum(axis=1)
        keep = n >= 30
        med = al.median(axis=1).where(keep)
        lo, hi = al.quantile(0.25, axis=1).where(keep), al.quantile(0.75, axis=1).where(keep)
        t = al.index / 60
        ax.fill_between(t, lo, hi, color=SERIES[0], alpha=0.12, linewidth=0)
        ax.plot(t, med.rolling(3, center=True).mean(), color=SERIES[0])
        ax.axvline(0, color=TEXT2, linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("hours from event")
    axes[0].set_ylabel("transpiration, ml/m²/h\n(median, IQR band)")
    fig.savefig(figdir / "activity_events.png")
    plt.close(fig)


def fig_price_profile(f: pd.DataFrame, figdir: Path) -> None:
    h = P.hourly_table(f, hourly_prices())
    h = h[h.index.month.isin([11, 12, 1, 2])]
    y = np.where(h.index.month >= 7, h.index.year, h.index.year - 1)
    h["season"] = [f"{a}/{str(a + 1)[2:]}" for a in y]
    t = h.groupby(["season", h.index.hour]).agg(lamp=("lamp", "mean"), price=("price", "mean"))
    counts = h.groupby("season").size()
    seasons = [s for s in counts.index if counts[s] > 24 * 60]
    cols = season_colors(seasons)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for s in seasons:
        g = t.loc[s]
        axes[0].plot(g.index, g["lamp"] * 100, color=cols[s], label=s)
        axes[1].plot(g.index, g["price"], color=cols[s], label=s)
    axes[0].set_title("Nov–Feb: share of the hour the lamps ran, %", fontsize=10)
    axes[1].set_title("Nov–Feb: day-ahead electricity price, €/MWh (DE-LU)", fontsize=10)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xlabel("hour of day (local)")
    axes[0].legend(fontsize=8, ncols=len(seasons), loc="lower left")
    fig.savefig(figdir / "price_profile.png")
    plt.close(fig)


def fig_frontier(out: Path, figdir: Path) -> None:
    """Weather-normalised heat against humid hours, per season (Oct-Apr, crop in)."""
    j = json.loads((out / "normalised.json").read_text())["heat"]["weather_only"]
    d = pd.read_csv(out / "daily.csv", index_col=0, parse_dates=True)
    d = d[
        (d.coverage > 0.9)
        & d.crop
        & d.month.isin([10, 11, 12, 1, 2, 3, 4])
        & (d.season >= j["base"])
    ]
    rh = d.groupby("season")["h_rh90"].mean()
    eff = {j["base"]: 0.0} | {s: e for s, (e, _) in j["effects"].items()}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for s in sorted(eff):
        x, y = rh[s], eff[s] / j["mean_heat"] * 100
        ax.scatter([x], [y], s=90, color=SEASON[s], edgecolor="white", linewidth=2, zorder=3)
        lab = s + (" (Oct–Nov)" if s == "2026/27" else "")
        ax.annotate(lab, (x, y), xytext=(8, 4), textcoords="offset points", color=TEXT, fontsize=9)
    ax.set_xlabel("hours per day with RH ≥ 90% (Oct–Apr mean)")
    ax.set_ylabel("heat vs 2023/24 at the same weather, %")
    ax.set_title("What dry air costs: heat against humid hours, per season", fontsize=10)
    ax.set_xlim(left=0)
    fig.savefig(figdir / "humidity_frontier.png")
    plt.close(fig)


def main() -> None:
    out, lg5, figdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    figdir.mkdir(parents=True, exist_ok=True)
    full = [s for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    f = F.five_min(pd.read_parquet(lg5))
    k = pd.read_csv(out / "kpis_season.csv", index_col=0)
    full = full or [s for s in k.index if k.loc[s, "days"] > 300]
    fig_seasons(out, figdir, full)
    fig_normalised(out, figdir)
    fig_frontier(out, figdir)
    fig_monthly_lamps(out, figdir)
    if (out / "planner.csv").exists():
        fig_planner(out / "planner.csv", figdir)
    fig_activity(f, figdir)
    fig_price_profile(f, figdir)
    print("figures in", figdir)


if __name__ == "__main__":
    main()

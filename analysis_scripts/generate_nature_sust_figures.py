r"""
Reproduce all five 600 DPI multi-panel manuscript figures from bundled experimental
outputs and empirical panels using relative paths (outputs PNG only).
"""
import json
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

ARCHIVE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ARCHIVE_ROOT / "data"
RAW_T2 = ARCHIVE_ROOT / "results" / "tier2_ultra"
RAW_T1 = ARCHIVE_ROOT / "results" / "tier1_standard"
MOD5_CAL = ARCHIVE_ROOT / "results" / "mod5a_calibrated_cb056"
FIG_DIR = ARCHIVE_ROOT / "reproduced_figures_600dpi"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ARCHIVE_ROOT / "engine"))
import cpu_reference as ref  # noqa: E402

# Global Nature Portfolio clean typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PAL = {
    "navy": "#1B365D",
    "blue": "#0072B2",
    "cyan": "#56B4E9",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "slate": "#5A6570",
    "black": "#111111",
    "gold": "#C69214",
}

PRETTY_PARAM = {
    "gamma_E": r"EROI drag ($\gamma_E$)",
    "epbt_base": r"Energy payback ($EPBT_0$)",
    "lambda_learn": r"Wright learning ($\lambda_{learn}$)",
    "tau_max": r"Fiscal cap ($\tau_{\max}$)",
    "gamma_C": r"Grid entropy ($\gamma_C$)",
    "c_max": r"Max consumption ($c_{\max}$)",
    "kappa_org": r"Coercion elasticity ($\kappa_{org}$)",
    "rho_c": r"Structural growth ($\rho_c$)",
    "delta_ent": r"Biophysical decay ($\delta_{ent}$)",
    "shock_scale": r"Stochastic shock ($\sigma_{sh}$)",
    "k_m_infra": r"Infra maintenance ($k_m$)",
    "k_t_age": r"Aging burden ($k_{age}$)",
    "k_cb_strain": r"Monetization strain ($k_{cb}$)",
    "gamma_D": r"Debt aversion ($\gamma_D$)",
    "gamma_O": r"Organizational drag ($\gamma_O$)",
    "psi_trade": r"Trade pass-through ($\psi_{tr}$)",
    "gamma_trade": r"Net-export sensitivity ($\gamma_{tr}$)",
    "eta_rec": r"Mineral recycling ($\eta_{rec}$)",
    "u_crit": r"Unemp barrier ($u_{crit}$)",
    "def_crit": r"Deficit barrier ($def_{crit}$)",
    "theta_rev": r"Legitimacy barrier ($\Theta_{rev}$)",
    "pi_crit": r"Inflation barrier ($\pi_{crit}$)",
}


def add_panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11.5, fontweight="bold",
            va="top", ha="left", color="#111111")


def load_all():
    ms_t2 = json.loads((RAW_T2 / "MASTER_SUMMARY.json").read_text(encoding="utf-8"))
    ms_t1 = json.loads((RAW_T1 / "MASTER_SUMMARY.json").read_text(encoding="utf-8"))
    m5_cal = json.loads((MOD5_CAL / "mod5_phase_ews_hyst_cb056.json").read_text(encoding="utf-8"))
    ds = ref.load_experiment_datasets(str(DATA_DIR))
    return ms_t2, ms_t1, m5_cal, ds


# =============================================================================
# FIGURE 1: 18-OECD Open Economy Regimes, Doughnut Scenarios & Falsification
# =============================================================================
def make_fig1(ms_t2, ds):
    m1 = ms_t2["modules"]["mod1"]
    z1 = np.load(RAW_T2 / "mod1_open18_arrays.npz")
    oc = ds["oecd"]
    yrs_hist = np.arange(1920, 2025)
    emp_g_yr = oc["obs_g"].mean(axis=0) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))
    plt.subplots_adjust(hspace=0.36, wspace=0.28)

    # (a) Historical trajectories vs empirical 18-OECD mean
    ax = axes[0, 0]
    add_panel_label(ax, "a")
    ax.plot(yrs_hist, emp_g_yr, color=PAL["black"], lw=2.2, label="Empirical 18-OECD mean", zorder=6)
    lines = [
        ("hist_1_Adaptive_Growth", "1 Adaptive Growth (93.8% viab)", PAL["blue"], "-", 1.9),
        ("hist_5B_PG_EcoSocial_Full", "5B Eco-Social Post-Growth (86.9%)", PAL["teal"], "--", 1.7),
        ("hist_2_Austerity_Coercion", "2 Coercive Austerity (0.0%)", PAL["vermilion"], "-.", 1.5),
        ("hist_3_Neoliberal_Cap", "3 Neoliberal Cap (0.02%)", PAL["orange"], ":", 1.6),
        ("hist_4_Gold_Standard", "4 Gold Standard (0.0%)", PAL["purple"], ":", 1.4),
    ]
    for k, lbl, col, ls, lw in lines:
        ax.plot(yrs_hist, z1[k][:, 0] * 100.0, color=col, ls=ls, lw=lw, label=lbl, alpha=0.92)
    ax.axvline(1973, color="#888888", ls="--", lw=0.8, alpha=0.7)
    ax.text(1974.2, 53.0, "1973 Oil &\nEROI Inflection", fontsize=7.2, color="#444444")
    g_imp = m1["historical_regimes"]["1_Adaptive_Growth"]["importer_g_2024_pct"]
    g_exp = m1["historical_regimes"]["1_Adaptive_Growth"]["exporter_g_2024_pct"]
    ax.text(0.03, 0.96, f"2024 Open-Economy Split (1_Adaptive):\nImporters: {g_imp:.1f}% vs Exporters: {g_exp:.1f}% (+{g_imp-g_exp:.1f} pp)",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.1,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F4F7FA", ec="#C6D2E0", lw=0.7))
    ax.set_title("Historical Public Expenditure Ratchet (1920–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Government Expenditure Share of GDP (%)")
    ax.set_xlim(1920, 2025)
    ax.set_ylim(10, 66)
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.9)

    # (b) 8 Institutional Regimes: Importer vs Exporter Viability
    ax = axes[0, 1]
    add_panel_label(ax, "b")
    reg_order = [
        ("1_Adaptive_Growth", "1 Adaptive Growth"),
        ("5B_PG_EcoSocial_Full", "5B PG Eco-Social"),
        ("5M_PG_Monetary_Only", "5M PG Monetary"),
        ("3_Neoliberal_Cap", "3 Neoliberal Cap"),
        ("5W_PG_WorkShare_Only", "5W PG WorkShare"),
        ("5A_PG_Unreformed", "5A PG Unreformed"),
        ("2_Austerity_Coercion", "2 Austerity"),
        ("4_Gold_Standard", "4 Gold Standard"),
    ]
    y_pos = np.arange(len(reg_order))
    imp_v = [m1["historical_regimes"][k]["importer_viability_pct"] for k, _ in reg_order]
    exp_v = [m1["historical_regimes"][k]["exporter_viability_pct"] for k, _ in reg_order]
    h = 0.36
    ax.barh(y_pos - h / 2, imp_v, height=h, color=PAL["vermilion"], alpha=0.88, label="Net Energy Importers (n=12)")
    ax.barh(y_pos + h / 2, exp_v, height=h, color=PAL["teal"], alpha=0.88, label="Net Energy Exporters (n=6)")
    for i, (iv, ev) in enumerate(zip(imp_v, exp_v)):
        avg_v = m1["historical_regimes"][reg_order[i][0]]["oecd_viability_pct"]
        xmax = max(iv, ev)
        ax.text(xmax + 1.4, y_pos[i], f"{avg_v:.1f}%", va="center", ha="left", fontsize=7.3, fontweight="bold", color="#333333")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in reg_order])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Century Viability Rate (% surviving 1920–2024)")
    ax.set_title("Institutional Complementarity Across 18 OECD Nations")
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD")

    # (c) 2025-2050 Doughnut Boundaries
    ax = axes[1, 0]
    add_panel_label(ax, "c")
    fwd = m1["forward_scenarios_2025_2050"]
    rect = patches.Rectangle((0, 0), 85.0, 1.0, facecolor="#DFF3E4", edgecolor=PAL["teal"], lw=1.3, ls="--", zorder=1)
    ax.add_patch(rect)
    ax.text(6.0, 0.12, "Safe & Just Doughnut Zone\n(CO2 <= 85 Gt, Minerals <= 1.0x)", fontsize=7.3, fontweight="bold", color="#0B6E4F", zorder=2)
    ax.axvline(85.0, color=PAL["teal"], ls="--", lw=1.0, alpha=0.75)
    ax.axhline(1.0, color=PAL["teal"], ls="--", lw=1.0, alpha=0.75)

    pts_cfg = [
        ("Scen1_Fossil_Decay", "Scen1 Fossil Decay", PAL["slate"], (-58, 12)),
        ("Scen2_Green_Growth_Elec", "Scen2 Green Growth", PAL["orange"], (10, 6)),
        ("Scen3_Breakthrough_Growth", "Scen3 Breakthrough", PAL["blue"], (10, -14)),
        ("Scen4a_PG_Unbuffered", "Scen4a PG Unbuffered", PAL["vermilion"], (8, 8)),
        ("Scen4b_PG_Circular_Doughnut", "Scen4b PG Circular Doughnut", PAL["teal"], (10, -16)),
    ]
    for sk, lbl, col, offset in pts_cfg:
        sv = fwd[sk]
        co2 = sv["total_18oecd_co2_2025_2050_gt"]
        mr = sv["mean_mineral_ceiling_ratio"]
        vb = sv["oecd_viability_pct"]
        sz = 70 + 3.2 * vb
        edge = "#111111" if sk == "Scen4b_PG_Circular_Doughnut" else "white"
        ax.scatter(co2, mr, s=sz, color=col, edgecolor=edge, linewidth=1.4, zorder=5)
        ax.annotate(
            f"{lbl}\n({vb:.1f}% viab | {co2:.1f} Gt | {mr:.2f}x)",
            xy=(co2, mr), xytext=offset, textcoords="offset points",
            fontsize=7.0, fontweight=("bold" if sk == "Scen4b_PG_Circular_Doughnut" else "normal"),
            color="#111111",
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6) if abs(offset[0]) > 15 or abs(offset[1]) > 10 else None,
            zorder=6,
        )
    ax.set_xlim(25, 315)
    ax.set_ylim(0.0, 1.48)
    ax.set_xlabel("Cumulative 18-OECD CO2 Emissions, 2025–2050 (GtCO2)")
    ax.set_ylabel("Peak Critical Mineral Demand / Supply Ceiling (ratio)")
    ax.set_title("2050 Planetary-Boundary & Macro-Viability Tradeoff")
    ax.grid(alpha=0.22, lw=0.5)

    # (d) Placebo Falsification & Blind Controller
    ax = axes[1, 1]
    add_panel_label(ax, "d")
    plac = m1["falsification_and_invariance"]["placebo_tests"]
    blind = m1["falsification_and_invariance"]["blind_no_age_controller"]
    cats = ["Full Physical\nBaseline", "Placebo 1:\nEROI Frozen (1920)", "Placebo 2:\nEROI & Age Frozen"]
    mae_full = [plac["baseline_MAE_full_pp"], plac["eroi_frozen_MAE_full_pp"], plac["eroi_and_age_frozen_MAE_full_pp"]]
    x = np.arange(len(cats))
    cols = [PAL["blue"], PAL["orange"], PAL["vermilion"]]
    bars = ax.bar(x, mae_full, width=0.50, color=cols, alpha=0.90, edgecolor="#222222", lw=0.7)
    for b, val in zip(bars, mae_full):
        ax.text(b.get_x() + b.get_width() / 2, val * 0.55, f"{val:.2f} pp\n(Full)", ha="center", va="center",
                fontsize=7.6, fontweight="bold", color="white")
    p70 = [plac["baseline_MAE_post1970_pp"], plac["eroi_frozen_MAE_post1970_pp"]]
    ax.plot([0, 1], p70, "o--", color=PAL["navy"], lw=1.6, ms=6, label="Post-1970 MAE (5.91 -> 7.40 pp)", zorder=6)
    ax.annotate("5.91 pp\n(Post-70)", xy=(0, p70[0]), xytext=(0, 6.75), ha="center", va="bottom",
                fontsize=7.1, fontweight="bold", color=PAL["navy"],
                arrowprops=dict(arrowstyle="-", color=PAL["navy"], lw=0.6))
    ax.annotate("7.40 pp\n(Post-70)", xy=(1, p70[1]), xytext=(1, 8.20), ha="center", va="bottom",
                fontsize=7.1, fontweight="bold", color=PAL["navy"],
                arrowprops=dict(arrowstyle="-", color=PAL["navy"], lw=0.6))
    ax.text(0.03, 0.95,
            f"Blind-No-Age Controller:\n  Viability = {blind['viability_pct']:.2f}%\n  Corr vs Emp G/Y = {blind['corr_vs_emp_g']:.2f}\n  g(2024) = {blind['g_2024_pct']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.1,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F4F7FA", ec="#C6D2E0", lw=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 11.2)
    ax.set_ylabel("Mean Absolute Error vs Historical 18-OECD G/Y (pp)")
    ax.set_title("Biophysical Placebo Falsification Tests")
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD")

    out = FIG_DIR / "Fig1_Open18_Regimes_and_Doughnut.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 2: Unbiased ABC-SMC, Survivorship Overlap, LOCO-CV & Cross-Tier
# =============================================================================
def make_fig2(ms_t2, ms_t1):
    m2 = ms_t2["modules"]["mod2"]
    m2_t1 = ms_t1["modules"]["mod2"]
    z2 = np.load(RAW_T2 / "mod2_abc_posterior.npz")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    plt.subplots_adjust(hspace=0.36, wspace=0.42)

    # (a) ABC-SMC Distance & Emergent Viability across rounds (Tier 2 + Tier 1)
    ax = axes[0, 0]
    add_panel_label(ax, "a")
    rs2 = m2["round_schedule"]
    rs1 = m2_t1["round_schedule"]
    r_x2 = [r["round"] for r in rs2]
    d_y2 = [r["median_dist"] for r in rs2]
    v_y2 = [r["emergent_viab_pct"] for r in rs2]
    r_x1 = [r["round"] for r in rs1]
    d_y1 = [r["median_dist"] for r in rs1]

    l1, = ax.plot(r_x2, d_y2, "o-", color=PAL["navy"], lw=2.0, ms=4.5, label="Median ABC distance (Tier 2, 18 rds)")
    l2, = ax.plot(r_x1, d_y1, "s--", color=PAL["cyan"], lw=1.4, ms=3.5, label="Median ABC distance (Tier 1, 14 rds)")
    ax.set_xlabel("ABC-SMC Sequential Round (zero viability penalty)")
    ax.set_ylabel("Median Fit Distance to Historical Panels", color=PAL["navy"])
    ax.tick_params(axis="y", labelcolor=PAL["navy"])
    ax.set_ylim(3.71, 3.98)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    l3, = ax2.plot(r_x2, v_y2, "^-", color=PAL["vermilion"], lw=1.9, ms=4.5, label="Unoptimized emergent viability (%)")
    ax2.set_ylabel("Emergent Century Viability (%)", color=PAL["vermilion"])
    ax2.tick_params(axis="y", labelcolor=PAL["vermilion"])
    ax2.set_ylim(92.0, 100.8)
    ax2.annotate(f"{v_y2[-1]:.2f}% emergent\n(Imp-Exp: {rs2[-1]['emergent_imp_exp_pp']:+.2f} pp)",
                 xy=(r_x2[-1], v_y2[-1]), xytext=(8.5, 95.3),
                 fontsize=7.2, fontweight="bold", color=PAL["vermilion"],
                 bbox=dict(boxstyle="round,pad=0.25", fc="#FFF7ED", ec=PAL["vermilion"], lw=0.7),
                 arrowprops=dict(arrowstyle="->", color=PAL["vermilion"], lw=0.9))
    ax.set_title("Unbiased ABC-SMC Convergence & Emergent Viability")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
              loc="lower left", frameon=True, framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.9)

    # (b) Survivorship-Conditioning Posterior Overlap (Normalized Quantiles)
    ax = axes[0, 1]
    add_panel_label(ax, "b", x=-0.16)
    post_full = z2["posterior_full"]
    p_viab = z2["posterior_emergent_viab"]
    sc = m2["survivorship_conditioning_check"]
    show_params = ["gamma_E", "epbt_base", "tau_max", "gamma_C", "c_max", "kappa_org", "k_m_infra", "delta_ent"]
    lo_arr = ref.ABC_PRIOR_BOUNDS[:, 0]
    hi_arr = ref.ABC_PRIOR_BOUNDS[:, 1]
    name_to_j = {ref.PARAM_NAMES[k_idx]: j for j, k_idx in enumerate(ref.ABC_ACTIVE_IDX)}
    mask95 = p_viab >= 0.95
    mask100 = p_viab >= (1.0 - 1e-9)

    y_p = np.arange(len(show_params))
    for i, pname in enumerate(show_params):
        j = name_to_j[pname]
        k_idx = ref.ABC_ACTIVE_IDX[j]
        lo, hi = lo_arr[j], hi_arr[j]
        u_q = (np.percentile(post_full[:, k_idx], [5, 50, 95]) - lo) / (hi - lo)
        c95_q = (np.percentile(post_full[mask95, k_idx], [5, 50, 95]) - lo) / (hi - lo)
        c100_q = (np.percentile(post_full[mask100, k_idx], [5, 50, 95]) - lo) / (hi - lo)
        ax.plot([u_q[0], u_q[2]], [y_p[i] - 0.18, y_p[i] - 0.18], color=PAL["navy"], lw=2.8, alpha=0.9, label="Unconditioned P(theta|Y)" if i == 0 else None)
        ax.scatter(u_q[1], y_p[i] - 0.18, color=PAL["navy"], s=24, zorder=5)
        ax.plot([c95_q[0], c95_q[2]], [y_p[i], y_p[i]], color=PAL["teal"], lw=2.3, alpha=0.9, label="Conditioned S >= 95% (99.96% ret.)" if i == 0 else None)
        ax.scatter(c95_q[1], y_p[i], color=PAL["teal"], marker="s", s=18, zorder=5)
        ax.plot([c100_q[0], c100_q[2]], [y_p[i] + 0.18, y_p[i] + 0.18], color=PAL["vermilion"], lw=1.8, ls="-", alpha=0.85, label="Strict S = 100% (58.74% ret.)" if i == 0 else None)
        ax.scatter(c100_q[1], y_p[i] + 0.18, color=PAL["vermilion"], marker="^", s=20, zorder=5)
    ax.set_yticks(y_p)
    ax.set_yticklabels([PRETTY_PARAM[p] for p in show_params])
    ax.set_ylim(len(show_params) - 0.3, -1.55)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Posterior [5th–95th Pct] Normalized to Uniform Prior [0, 1]")
    ax.set_title("Survivorship-Conditioned Posterior Equivalence")
    s95 = sc["viab_ge_95pct"]
    s100 = sc["viab_eq_100pct"]
    ax.text(0.02, 0.03,
            f"S>=95%: CI95 overlap={s95['mean_ci95_jaccard_pct']:.2f}%, Shift={s95['mean_rel_median_shift_pct']:.3f}%\n"
            f"S=100%: CI95 overlap={s100['mean_ci95_jaccard_pct']:.2f}%, Shift={s100['mean_rel_median_shift_pct']:.2f}%",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F4F7FA", ec="#C6D2E0", lw=0.7))
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.95, edgecolor="#DDDDDD", fontsize=6.7)

    # (c) 18-Country Blind LOCO-CV
    ax = axes[1, 0]
    add_panel_label(ax, "c")
    loco_sorted = sorted(m2["loco_cv"].items(), key=lambda kv: kv[1]["blind_oos_r"])
    isos_s = [k for k, _ in loco_sorted]
    r_s = [v["blind_oos_r"] for _, v in loco_sorted]
    mae_s = [v["blind_oos_mae_pp"] for _, v in loco_sorted]
    y_c = np.arange(len(isos_s))
    bar_cols = [PAL["teal"] if rv >= 0.80 else (PAL["blue"] if rv >= 0.70 else PAL["orange"]) for rv in r_s]
    ax.barh(y_c, r_s, height=0.68, color=bar_cols, alpha=0.88)
    med_r = float(np.median(r_s))
    ax.axvline(med_r, color=PAL["vermilion"], ls="--", lw=1.3, label=f"Median LOCO R = {med_r:.3f}")
    for i, (rv, mv) in enumerate(zip(r_s, mae_s)):
        ax.text(rv + 0.012, y_c[i], f"{rv:.2f} ({mv:.1f} pp)", va="center", ha="left", fontsize=6.7, color="#222222")
    ax.set_yticks(y_c)
    ax.set_yticklabels(isos_s, fontsize=7.2)
    ax.set_xlim(0, 1.24)
    ax.set_xlabel("Blind Out-of-Sample Correlation R with Historical G/Y (and MAE)")
    ax.set_title("Leave-One-Country-Out Cross-Validation (All 18 OECD)")
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD")

    # (d) Cross-Tier Independent Posterior Convergence Parity
    ax = axes[1, 1]
    add_panel_label(ax, "d")
    p1 = m2_t1["parameter_posteriors"]
    p2 = m2["parameter_posteriors"]
    n1 = np.array([(p1[ref.PARAM_NAMES[idx]]["median"] - lo_arr[j]) / (hi_arr[j] - lo_arr[j]) for j, idx in enumerate(ref.ABC_ACTIVE_IDX)])
    n2 = np.array([(p2[ref.PARAM_NAMES[idx]]["median"] - lo_arr[j]) / (hi_arr[j] - lo_arr[j]) for j, idx in enumerate(ref.ABC_ACTIVE_IDX)])
    r_cross = float(np.corrcoef(n1, n2)[0, 1])
    max_diff = float(np.max(np.abs(n1 - n2)))

    ax.plot([0, 1], [0, 1], "--", color="#666666", lw=1.2, label="1:1 Exact Parity")
    ax.scatter(n1, n2, s=55, color=PAL["blue"], edgecolor="white", lw=0.8, zorder=5, label="16 Active ABC Parameters")
    highlight = {
        "gamma_E": (12, -8),
        "epbt_base": (12, 8),
        "tau_max": (-85, 18),
        "k_m_infra": (-92, -18),
        "c_max": (-48, 16),
        "gamma_C": (-30, -22),
    }
    for j, idx in enumerate(ref.ABC_ACTIVE_IDX):
        nm = ref.PARAM_NAMES[idx]
        if nm in highlight:
            ax.annotate(nm, xy=(n1[j], n2[j]), xytext=highlight[nm], textcoords="offset points",
                        fontsize=7.2, fontweight="bold", color="#111111",
                        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6))
    ax.text(0.04, 0.94, f"Tier 1 (14-rd) vs Tier 2 (18-rd)\nPearson R = {r_cross:.5f}\nMax norm. diff = {max_diff:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F4F7FA", ec="#C6D2E0", lw=0.7))
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Tier 1 Standard (14 Rounds) Normalized Posterior Median")
    ax.set_ylabel("Tier 2 Ultra (18 Rounds) Normalized Posterior Median")
    ax.set_title("Cross-Scale Bayesian Posterior Convergence Check")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD")

    out = FIG_DIR / "Fig2_Unbiased_ABC_Survivorship_and_LOCO.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 3: 21-Group x 22-Parameter Saltelli-Jansen Sobol Global Sensitivity
# =============================================================================
def make_fig3(ms_t2):
    m3 = ms_t2["modules"]["mod3"]
    z3 = np.load(RAW_T2 / "mod3_sobol_arrays.npz")
    pnames = [str(x) for x in z3["param_names"]]

    # Top 10 parameters by OECD_Mean years_alive S_Ti
    oecd_sti = np.array([m3["groups"]["OECD_Mean"]["years_alive"]["S_Ti"][p] for p in pnames])
    top10_idx = np.argsort(-oecd_sti)[:10]
    top10_params = [pnames[i] for i in top10_idx]

    group_order = [
        "OECD_Mean", "Importer_Mean", "Exporter_Mean",
        "AUS", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA",
        "GBR", "IRL", "ITA", "JPN", "NLD", "NOR", "PRT", "SWE", "USA"
    ]
    mat = np.zeros((len(group_order), len(top10_params)), dtype=np.float64)
    for gi, gname in enumerate(group_order):
        g_sti = m3["groups"][gname]["years_alive"]["S_Ti"]
        for pi, pname in enumerate(top10_params):
            mat[gi, pi] = g_sti[pname]

    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(14.4, 7.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    plt.subplots_adjust(wspace=0.52)

    # (a) 21-Group Heatmap
    add_panel_label(ax_h, "a", x=-0.12, y=1.04)
    cmap = LinearSegmentedColormap.from_list("nat_heat", ["#F8FAFC", "#B8D8EB", "#2C7FB8", "#1B365D"])
    im = ax_h.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.40)
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.030, pad=0.02)
    cbar.set_label("Sobol Total-Effect Index ($S_{Ti}$)", fontsize=7.8)

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            txt_col = "white" if val >= 0.22 else "#111111"
            fw = "bold" if (r == 15 and c == 3) or val >= 0.20 else "normal"
            ax_h.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6.6, color=txt_col, fontweight=fw)

    # Highlight JPN tau_max singularity (row index 15 = JPN, col index 3 = tau_max)
    jpn_r = group_order.index("JPN")
    tau_c = top10_params.index("tau_max")
    rect_jpn = patches.Rectangle((tau_c - 0.5, jpn_r - 0.5), 1.0, 1.0, fill=False, edgecolor=PAL["vermilion"], lw=2.2, zorder=6)
    ax_h.add_patch(rect_jpn)
    ax_h.axhline(2.5, color="#111111", lw=1.3, ls="--")

    ax_h.set_xticks(np.arange(len(top10_params)))
    ax_h.set_xticklabels([PRETTY_PARAM[p].split(" (")[0] + "\n(" + PRETTY_PARAM[p].split(" (")[1] for p in top10_params],
                         rotation=35, ha="right", fontsize=7.0)
    ax_h.set_yticks(np.arange(len(group_order)))
    y_lbls = [f"{g}*" if g == "JPN" else g for g in group_order]
    ax_h.set_yticklabels(y_lbls, fontsize=7.5)
    for tick, g in zip(ax_h.get_yticklabels(), group_order):
        if g == "JPN":
            tick.set_color(PAL["vermilion"])
            tick.set_fontweight("bold")
    ax_h.set_title("21-Group × 22-Parameter Sobol Sensitivity (>2.1B Trajectories)")

    # (b) OECD_Mean Si vs STi + Bootstrap 95% CIs + CO2/Mineral drivers inset
    add_panel_label(ax_b, "b", x=-0.20, y=1.04)
    ya = m3["groups"]["OECD_Mean"]["years_alive"]
    si_vals = np.array([ya["S_i"][p] for p in top10_params])
    sti_vals = np.array([ya["S_Ti"][p] for p in top10_params])
    lo_err = np.array([ya["S_Ti"][p] - ya["S_Ti_ci95"][p][0] for p in top10_params])
    hi_err = np.array([ya["S_Ti_ci95"][p][1] - ya["S_Ti"][p] for p in top10_params])

    y_pos = np.arange(len(top10_params))
    h = 0.36
    ax_b.barh(y_pos + h / 2, sti_vals, height=h, color=PAL["navy"], alpha=0.92, label="Total effect $S_{Ti}$ (with 95% bootstrap CI)")
    ax_b.errorbar(sti_vals, y_pos + h / 2, xerr=[lo_err, hi_err], fmt="none", ecolor="#111111", elinewidth=1.0, capsize=2.2, zorder=5)
    ax_b.barh(y_pos - h / 2, si_vals, height=h, color=PAL["cyan"], alpha=0.92, label="Main (first-order) effect $S_i$")

    for i, (sv, stv) in enumerate(zip(si_vals, sti_vals)):
        ax_b.text(stv + hi_err[i] + 0.008, y_pos[i] + h / 2, f"{stv:.3f}", va="center", ha="left", fontsize=7.2, fontweight="bold", color=PAL["navy"])
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([PRETTY_PARAM[p] for p in top10_params], fontsize=7.5)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 0.48)
    ax_b.set_xlabel("Sobol Sensitivity Index (OECD_Mean, N=65,536, CRN=64)")
    ax_b.set_title("Biophysical Dominance & Interaction Structure")
    ax_b.grid(axis="x", alpha=0.22, lw=0.5)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD", fontsize=7.0)

    # Clean box summarizing CO2 & Mineral Top-3 Drivers + JPN singularity (safely right of short bottom bars)
    co2_top = sorted(m3["groups"]["OECD_Mean"]["co2_cum_2050"]["S_Ti"].items(), key=lambda x: -x[1])[:3]
    min_top = sorted(m3["groups"]["OECD_Mean"]["mineral_ratio_2050"]["S_Ti"].items(), key=lambda x: -x[1])[:3]
    summary_txt = (
        "CO2 (2050) Top STi:\n  " + ", ".join(f"{k}={v:.2f}" for k, v in co2_top) + "\n"
        "Critical Minerals Top STi:\n  " + ", ".join(f"{k}={v:.2f}" for k, v in min_top) + "\n"
        "*JPN Singularity: tau_max (0.178)\n  displaces lambda_learn (0.159)"
    )
    ax_b.text(0.98, 0.03, summary_txt, transform=ax_b.transAxes, va="bottom", ha="right", fontsize=6.7,
              bbox=dict(boxstyle="round,pad=0.35", fc="#FFF7ED", ec=PAL["vermilion"], lw=0.9))

    out = FIG_DIR / "Fig3_Sobol_21Group_Biophysical_Triad.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 4: Calibrated Phase Cliff, EWS CSD, Hysteresis & 4D Mineral-Cannib
# =============================================================================
def make_fig4(ms_t2, m5_cal):
    z5 = np.load(MOD5_CAL / "mod5_phase_ews_hyst_cb056.npz")
    z4 = np.load(RAW_T2 / "mod4_cannib_minerals.npz")

    ea = z5["eroi_axis"]
    ga = z5["gorg_axis"] * 100.0
    imp_m = z5["Importer_Net"]
    exp_m = z5["Exporter_Net"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    plt.subplots_adjust(hspace=0.36, wspace=0.44)

    # (a) 512x512 Calibrated Phase Cliff
    ax = axes[0, 0]
    add_panel_label(ax, "a")
    im = ax.imshow(imp_m[:, :, 0] * 100.0, origin="lower", aspect="auto",
                   extent=[ea[0], ea[-1], ga[0], ga[-1]], cmap="RdYlBu", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Viability (%) — Importer_Net", fontsize=7.6)

    EE, GG = np.meshgrid(ea, ga, indexing="xy")
    ax.contour(EE, GG, imp_m[:, :, 0], levels=[0.50], colors=["white"], linewidths=2.0, linestyles="-")
    ax.contour(EE, GG, exp_m[:, :, 0], levels=[0.50], colors=[PAL["cyan"]], linewidths=1.8, linestyles="--")

    s_imp = m5_cal["phase_and_ews_slices"]["Importer_Net"]
    s_exp = m5_cal["phase_and_ews_slices"]["Exporter_Net"]
    ax.scatter([s_imp["min_viable_eroi_at_gorg_5pct"], s_imp["min_viable_eroi_at_gorg_10pct"]], [5.0, 10.0],
               color="white", edgecolor="#111111", s=45, zorder=6)
    ax.annotate(f"g_org=5%:\nImp {s_imp['min_viable_eroi_at_gorg_5pct']:.2f} vs Exp {s_exp['min_viable_eroi_at_gorg_5pct']:.2f}",
                xy=(s_imp["min_viable_eroi_at_gorg_5pct"], 5.0), xytext=(15.5, 4.0),
                fontsize=7.0, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#444444", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#111111", lw=0.9))
    ax.annotate(f"g_org=10%:\nImp {s_imp['min_viable_eroi_at_gorg_10pct']:.2f} vs Exp {s_exp['min_viable_eroi_at_gorg_10pct']:.2f}",
                xy=(s_imp["min_viable_eroi_at_gorg_10pct"], 10.0), xytext=(18.5, 10.2),
                fontsize=7.0, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#444444", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#111111", lw=0.9))
    ax.text(5.2, 15.6, "Collapsed Coercive\nZone (0% Viable)", color="white", fontsize=7.8, fontweight="bold")
    ax.text(22.2, 6.2, "Solid white: Imp 50%\nDashed cyan: Exp 50%", color="#111111", fontsize=6.9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88, ec="none"))
    ax.set_xlabel("System EROI Ratio")
    ax.set_ylabel("Coercive Control Expenditure $g_{org}$ (% of GDP)")
    ax.set_title("512×512 Viability Cliff ($c_b=0.560$ Calibrated Corridor)")

    # (b) Pre-Collapse EWS Critical Slowing Down Transect along g_org = 8%
    ax = axes[0, 1]
    add_panel_label(ax, "b")
    ig8 = int(np.argmin(np.abs(ga - 8.0)))
    surv_tr = imp_m[ig8, :, 0] * 100.0
    var_tr = imp_m[ig8, :, 3] * 1e4  # x 1e-4 for clean scale
    ar1d_tr = imp_m[ig8, :, 2]

    mask_tip = (surv_tr >= 15.0) & (surv_tr <= 85.0)
    if np.any(mask_tip):
        e_lo_t, e_hi_t = ea[mask_tip].min(), ea[mask_tip].max()
        ax.axvspan(e_lo_t, e_hi_t, color="#FFE8D6", alpha=0.65, label="Pre-collapse tipping band")
    l_s, = ax.plot(ea, surv_tr, color=PAL["navy"], lw=2.1, label="Viability (%) at $g_{org}=8\\%$")
    ax.set_xlabel("System EROI Ratio (transect along $g_{org} = 8\\%$)")
    ax.set_ylabel("Viability Rate (%)", color=PAL["navy"])
    ax.tick_params(axis="y", labelcolor=PAL["navy"])
    ax.set_xlim(6, 26)
    ax.set_ylim(-4, 108)

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    l_v, = ax_r.plot(ea, var_tr, color=PAL["vermilion"], lw=1.9, label="Infra variance $Var(B_{infra}) \\times 10^{-4}$")
    l_a, = ax_r.plot(ea, np.maximum(ar1d_tr, 0.0) * 6.0, "--", color=PAL["teal"], lw=1.5, label="$AR(1)(\\Delta d_{tot}) \\times 6$")
    ax_r.set_ylabel("Early-Warning Intensity (CSD Variance / AR(1))", color=PAL["vermilion"])
    ax_r.tick_params(axis="y", labelcolor=PAL["vermilion"])
    ax_r.set_ylim(-0.2, 6.2)
    ax.text(0.97, 0.94,
            f"Slice mean CSD surge (Safe -> Tipping):\nVar(B_infra): {s_imp['ews_var_binfra_safe']:.5f} -> {s_imp['ews_var_binfra_tipping']:.5f} (6.7x)\nAR(1)(d_tot): {s_imp['ews_ar1_diff_debt_safe']:.3f} -> {s_imp['ews_ar1_diff_debt_tipping']:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=6.9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF7ED", ec=PAL["vermilion"], lw=0.7))
    ax.set_title("Critical Slowing Down Prior to Thermodynamic Collapse")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend([l_s, l_v, l_a], [l_s.get_label(), l_v.get_label(), l_a.get_label()],
              loc="lower left", frameon=True, framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.8)

    # (c) Mod 5B Irreversible Hysteresis Loop (RTX 5080 32,768 seeds)
    ax = axes[1, 0]
    add_panel_label(ax, "c")
    eh = z5["eroi_hyst"]
    m5h = ms_t2["modules"]["mod5"]["hysteresis_loop"]
    h_cfg = [
        ("hyst_1_Adaptive_Growth", "1_Adaptive_Growth", "1 Adaptive Growth (99.98% rec.)", PAL["blue"]),
        ("hyst_2_Austerity_Coercion", "2_Austerity_Coercion", "2 Austerity Coercion (0.0% rec.)", PAL["vermilion"]),
        ("hyst_3_Neoliberal_Cap", "3_Neoliberal_Cap", "3 Neoliberal Cap (0.0% rec.)", PAL["orange"]),
    ]
    for arr_k, j_k, lbl, col in h_cfg:
        tr = z5[arr_k]
        # col 1 is O_state; first 40 steps = EROI 25->6, next 40 = 6->25
        ax.plot(eh[:40], tr[:40, 1], color=col, ls="-", lw=2.0, label=lbl)
        ax.plot(eh[39:], tr[39:, 1], color=col, ls="--", lw=1.7, alpha=0.9)
    # Direction arrows on Adaptive vs Neoliberal
    tr1 = z5["hyst_1_Adaptive_Growth"]
    tr3 = z5["hyst_3_Neoliberal_Cap"]
    ax.annotate("", xy=(14.5, tr1[18, 1]), xytext=(16.5, tr1[15, 1]),
                arrowprops=dict(arrowstyle="->", color=PAL["blue"], lw=1.5))
    ax.annotate("", xy=(16.5, tr3[57, 1]), xytext=(14.5, tr3[54, 1]),
                arrowprops=dict(arrowstyle="->", color=PAL["orange"], lw=1.5))
    ax.text(0.03, 0.94, "Solid: Forward EROI decay (25 -> 6)\nDashed: EROI recovery (6 -> 25)",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F4F7FA", ec="#C6D2E0", lw=0.7))
    ax.set_xlim(5.2, 26.0)
    ax.set_ylim(0.15, 0.88)
    ax.set_xlabel("Exogenous EROI Shock–Recovery Cycle (25 -> 6 -> 25)")
    ax.set_ylabel("Structural Coordination Capacity $O_{state}$")
    ax.set_title("Irreversible Path Hysteresis Under EROI Shock (32,768 Seeds)")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD", fontsize=7.1)

    # (d) Mod 4 4D Energy Cannibalization x Critical Minerals x Post-Growth
    ax = axes[1, 1]
    add_panel_label(ax, "d")
    agg4 = z4["agg_4d"]  # (2, 24, 20, 16, 6)
    rec_ax = z4["rec_vals"] * 100.0
    lam_ax = z4["lam_vals"]
    best_nf_i = int(np.unravel_index(np.argmax(agg4[1, :, :, :, 5]), agg4[1, :, :, :, 5].shape)[0])
    nf_best = z4["nonf_tgts"][best_nf_i] * 100.0
    pg_safe_slice = agg4[1, best_nf_i, :, :, 5] * 100.0  # (rec=20, lam=16)
    pg_mr_slice = agg4[1, best_nf_i, :, :, 2]

    LL, RR = np.meshgrid(lam_ax, rec_ax, indexing="xy")
    im4 = ax.imshow(pg_mr_slice, origin="lower", aspect="auto",
                    extent=[lam_ax[0], lam_ax[-1], rec_ax[0], rec_ax[-1]], cmap="RdYlGn_r", vmin=0.65, vmax=1.32)
    cbar4 = fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.03)
    cbar4.set_label("Mineral Demand Ceiling Ratio (× supply)", fontsize=7.5)
    cs_mr = ax.contour(LL, RR, pg_mr_slice, levels=[0.8, 0.9, 1.0, 1.15],
                       colors=["#0B4F3A", "#0B4F3A", PAL["navy"], "#444444"], linewidths=[1.1, 1.1, 2.0, 1.0],
                       linestyles=[":", ":", "-", "--"])
    ax.clabel(cs_mr, fmt={0.8: "0.8x", 0.9: "0.9x", 1.0: "Mineral = 1.0x", 1.15: "1.15x"}, fontsize=7.0)
    # Highlight narrow Triple-Safe corridor (Post-Growth safe_and_just > 0%)
    ax.contourf(LL, RR, pg_safe_slice, levels=[0.5, 15.0], hatches=["///"], colors="none")
    ax.contour(LL, RR, pg_safe_slice, levels=[0.5], colors=[PAL["black"]], linewidths=1.8, linestyles="-")
    ax.text(0.618, 66.0, "Triple-Safe\nZone\n(10.55%\npeak)", fontsize=7.1, fontweight="bold", color="#111111",
            ha="center", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=PAL["black"], alpha=0.88))

    m4 = ms_t2["modules"]["mod4"]
    ax.text(0.97, 0.04,
            f"Slice at optimal non-fossil = {nf_best:.1f}% (CO2=63.5 Gt)\n"
            f"4D Grid (15.73M traj) Safe Corridor:\n"
            f"  Growth max = {m4['max_triple_boundary_safe_just_growth_pct']:.1f}% (min CO2 109.1 Gt)\n"
            f"  Post-Growth max = {m4['max_triple_boundary_safe_just_postgrowth_pct']:.2f}% (min CO2 58.5 Gt)",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=6.7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#444444", alpha=0.92))
    ax.set_xlabel("Wright Manufacturing Learning Multiplier $\\lambda_{learn}$ (lower = faster)")
    ax.set_ylabel("Critical Mineral Recycling Efficiency $\\eta_{rec}$ (%)")
    ax.set_title("4D Energy Cannibalization × Mineral Safe Envelope")
    ax.grid(alpha=0.15, lw=0.4)

    out = FIG_DIR / "Fig4_Phase_Cliff_EWS_Hysteresis_and_4D_Doughnut.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 5: China Zero-Leakage 5-Way Factorial, Robustness & 24x24 Frontier
# =============================================================================
def make_fig5(ms_t2, ds):
    m6 = ms_t2["modules"]["mod6"]
    z6 = np.load(RAW_T2 / "mod6_china_oos.npz")
    ch = ds["china"]
    yrs = z6["years"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    plt.subplots_adjust(hspace=0.36, wspace=0.44)

    # (a) Blind Historical Validation + 2050 Fan Charts (Public Debt & G/Y)
    ax = axes[0, 0]
    add_panel_label(ax, "a")
    obs_d = ch["obs_debt"] * 100.0
    obs_g = ch["obs_g"] * 100.0
    yrs75 = yrs[:75]
    vd = ~np.isnan(obs_d)
    vg = ~np.isnan(obs_g)

    # Debt on primary y-axis
    d_d = z6["CHN_D_Current_Elec_Infra_Coord_d_pcts"] * 100.0
    d_c = z6["CHN_C_Sovereign_FossilInertia_d_pcts"] * 100.0
    ax.fill_between(yrs, d_d[0], d_d[4], color=PAL["blue"], alpha=0.14)
    l_dd, = ax.plot(yrs, d_d[2], color=PAL["blue"], lw=2.0, label="Blind Sim Debt/GDP: CHN_D (p5–p95 band)")
    l_dc, = ax.plot(yrs, d_c[2], "--", color=PAL["vermilion"], lw=1.6, label="Blind Sim Debt/GDP: CHN_C (Fossil)")
    l_od = ax.scatter(yrs75[vd], obs_d[vd], color=PAL["navy"], s=14, zorder=6, label="IMF WEO 2025-04 Public Debt/GDP")

    # G/Y on twin axis shifted nicely or clean secondary axis
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    g_d = z6["CHN_D_Current_Elec_Infra_Coord_g_pcts"] * 100.0
    l_gd, = ax2.plot(yrs, g_d[2], color=PAL["teal"], lw=1.8, label="Blind Sim Fiscal Share G/Y: CHN_D")
    l_og = ax2.scatter(yrs75[vg], obs_g[vg], marker="^", color="#0B6E4F", s=16, alpha=0.85, zorder=6, label="IMF WEO General Gov. Expenditure/GDP")

    ax.axvline(2024, color="#555555", ls=":", lw=1.1)
    ax.text(2025.5, 128, "2025–2050\nFactorial Switch", fontsize=7.0, color="#444444")
    bf = m6["blind_historical_fit"]
    ax.text(0.03, 0.95,
            f"ZERO China Data Calibration:\n"
            f"  Debt R (1995–2024) = {bf['blind_debt_corr']:.3f}\n"
            f"  2024 Debt: {bf['blind_sim_debt_2024_med_pct']:.1f}% vs IMF {bf['obs_debt_2024_pct']:.1f}%\n"
            f"  2024 G/Y: {bf['blind_sim_g_2024_med_pct']:.1f}% vs IMF {bf['obs_g_2024_pct']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F4F7FA", ec="#C6D2E0", lw=0.7))
    ax.set_xlim(1950, 2051)
    ax.set_ylim(0, 145)
    ax2.set_ylim(0, 95)
    ax.set_xlabel("Year")
    ax.set_ylabel("Public Debt / GDP (%)", color=PAL["navy"])
    ax2.set_ylabel("Fiscal Share G / GDP (%)", color=PAL["teal"])
    ax.set_title("Blind Historical Validation (1950–2024) & Projections")
    ax.grid(alpha=0.20, lw=0.5)
    hs = [l_dd, l_dc, l_od, l_gd, l_og]
    ax.legend(hs, [h.get_label() for h in hs], loc="upper left", bbox_to_anchor=(0.01, 0.72), frameon=True, framealpha=0.90, fontsize=6.4)

    # (b) 5-Way Factorial 3-Metric Dashboard
    ax = axes[0, 1]
    add_panel_label(ax, "b", x=-0.16)
    f5 = m6["controlled_5way_factorial"]
    p_order = [
        ("CHN_A_Austerity_SlowElec", "CHN_A Austerity +\nSlow Elec"),
        ("CHN_B_FastElec_Under_Austerity", "CHN_B Austerity +\nFast Elec"),
        ("CHN_C_Sovereign_FossilInertia", "CHN_C Sovereign +\nFossil Inertia"),
        ("CHN_D_Current_Elec_Infra_Coord", "CHN_D Current Elec\n& Infra Coord"),
        ("CHN_E_Elec_Plus_EcoSocial_Rebalance", "CHN_E Current + Eco-\nSocial & Recycling"),
    ]
    y_p = np.arange(len(p_order))
    viab_v = [f5[k]["viability_2050_pct"] for k, _ in p_order]
    co2_v = [f5[k]["co2_cum_2025_2050_med_gt"] for k, _ in p_order]
    mr_v = [f5[k]["mineral_ratio_med"] for k, _ in p_order]

    cols_5 = [PAL["slate"], PAL["orange"], PAL["vermilion"], PAL["blue"], PAL["teal"]]
    ax.barh(y_p, viab_v, height=0.58, color=cols_5, alpha=0.90, edgecolor="#222222", lw=0.7)
    for i, (vb, c2, mv) in enumerate(zip(viab_v, co2_v, mr_v)):
        co2_tag = f"{c2:.1f} Gt" + (" (OK)" if c2 <= 140.0 else " (>140)")
        min_tag = f"{mv:.2f}x" + (" (OK)" if mv <= 1.0 else " (>1.0x)")
        ax.text(vb + 2.2, y_p[i], f"{vb:.2f}%  |  CO2: {co2_tag}  |  Min: {min_tag}",
                va="center", ha="left", fontsize=6.7,
                fontweight=("bold" if i >= 3 else "normal"), color="#111111")
    ax.set_yticks(y_p)
    ax.set_yticklabels([lbl for _, lbl in p_order], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 205)
    ax.set_xlabel("2050 Macro-Financial Viability (%) & Boundary Diagnostics")
    ax.set_title("Zero-Leakage Symmetric 5-Way Factorial Outcomes")
    ax.axvline(95, color="#888888", ls="--", lw=0.8)
    ax.grid(axis="x", alpha=0.22, lw=0.5)

    # (c) Wright Learning Sensitivity & 3-Spec Structural Invariance
    ax = axes[1, 0]
    add_panel_label(ax, "c")
    wl = m6["wright_learning_sensitivity_CHN_D"]
    si_chn = m6["structural_invariance"]
    lams = ["lam_0.72", "lam_0.85", "lam_0.95"]
    lam_labels = ["Fast\n($\\lambda=0.72$)", "Moderate\n($\\lambda=0.85$)", "Conserv.\n($\\lambda=0.95$)"]
    w_viab = [wl[k]["viability_pct"] for k in lams]
    w_min = [wl[k]["mineral_ratio_med"] for k in lams]

    x_pos = np.array([0.0, 1.0, 2.0])
    b_v = ax.bar(x_pos, w_viab, width=0.50, color=PAL["blue"], alpha=0.88, label="CHN_D Viability (%)")
    for b, vv in zip(b_v, w_viab):
        ax.text(b.get_x() + b.get_width() / 2, vv + 1.4, f"{vv:.2f}%", ha="center", va="bottom", color=PAL["navy"], fontsize=7.0, fontweight="bold")
    ax.set_ylim(0, 122)
    ax.set_ylabel("2050 Viability (%)")

    ta_keys = [("high_trade_expose_ta0.50_psi0.048", "High Trade\n($\\theta_a=0.50$)"),
               ("baseline_ta0.75_psi0.032", "Baseline\n($\\theta_a=0.75$)"),
               ("low_trade_expose_ta0.90_psi0.015", "Low Trade\n($\\theta_a=0.90$)")]
    x_tr = np.array([3.5, 4.5, 5.5])
    ax.plot(x_tr, [si_chn[k]["CHN_D_Current_Elec_Infra_Coord"] for k, _ in ta_keys], "o-", color=PAL["blue"], lw=1.8, ms=6, label="Trade invar.: CHN_D (100.0%)")
    ax.plot(x_tr, [si_chn[k]["CHN_E_Elec_Plus_EcoSocial_Rebalance"] for k, _ in ta_keys], "s--", color=PAL["teal"], lw=1.8, ms=5.5, label="Trade invar.: CHN_E (97.9–98.1%)")
    ax.plot(x_tr, [si_chn[k]["CHN_A_Austerity_SlowElec"] for k, _ in ta_keys], "^:", color=PAL["vermilion"], lw=1.6, ms=5.5, label="Trade invar.: CHN_A (0.0%)")

    ax.axvline(2.75, color="#888888", ls="--", lw=1.0)
    ax.text(1.0, 114, "Wright Learning (CHN_D)", ha="center", fontsize=7.3, fontweight="bold", color=PAL["navy"])
    ax.text(4.5, 114, "Trade-Closure Invariance", ha="center", fontsize=7.3, fontweight="bold", color=PAL["navy"])

    ax2_c = ax.twinx()
    ax2_c.spines["top"].set_visible(False)
    l_mr, = ax2_c.plot(x_pos, w_min, "D-", color=PAL["vermilion"], lw=2.0, ms=5.5, zorder=6, label="CHN_D Mineral Ratio (right)")
    for xv, mv in zip(x_pos, w_min):
        ax2_c.text(xv, mv + 0.08, f"{mv:.2f}x", ha="center", fontsize=7.1, fontweight="bold", color=PAL["vermilion"],
                   bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82))
    ax2_c.set_ylim(0.5, 2.55)
    ax2_c.set_ylabel("CHN_D Mineral Demand Ratio (× supply)", color=PAL["vermilion"])
    ax2_c.tick_params(axis="y", labelcolor=PAL["vermilion"])

    ax.set_xticks(list(x_pos) + list(x_tr))
    ax.set_xticklabels(lam_labels + [lbl for _, lbl in ta_keys], fontsize=6.9)
    ax.set_title("Sensitivity to Learning Speed & Trade Closure Assumptions")
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    h1, lb1 = ax.get_legend_handles_labels()
    ax.legend(h1 + [l_mr], lb1 + [l_mr.get_label()], loc="center right", frameon=True, framealpha=0.93, edgecolor="#DDDDDD", fontsize=6.4)

    # (d) 24x24 Policy Synergy Frontier (Recycling x Social Transfer Ceiling)
    ax = axes[1, 1]
    add_panel_label(ax, "d", x=-0.16)
    rec_ax = z6["chn_frontier_rec_ax"] * 100.0
    gt_ax = z6["chn_frontier_gt_ax"] * 100.0
    f_surv = z6["chn_frontier_surv"]      # (24 rec, 24 gt): 100.0% across all 576 cells!
    f_min = z6["chn_frontier_minratio"]   # (24 rec, 24 gt): 0.976x -> 1.308x

    GGt, RRc = np.meshgrid(gt_ax, rec_ax, indexing="xy")
    im6 = ax.imshow(f_min, origin="lower", aspect="auto",
                    extent=[gt_ax[0], gt_ax[-1], rec_ax[0], rec_ax[-1]], cmap="RdYlGn_r", vmin=0.95, vmax=1.32)
    cbar6 = fig.colorbar(im6, ax=ax, fraction=0.046, pad=0.03)
    cbar6.set_label("Peak Critical Mineral Ratio (× supply)", fontsize=7.5)

    cs_min = ax.contour(GGt, RRc, f_min, levels=[1.0, 1.1, 1.2],
                        colors=["#0B4F3A", "#333333", "#333333"], linewidths=[2.2, 1.1, 1.0], linestyles=["-", "--", ":"])
    ax.clabel(cs_min, fmt={1.0: "Mineral = 1.0x", 1.1: "1.1x", 1.2: "1.2x"}, fontsize=7.2, manual=[(19.0, 81.2), (26.5, 64.5), (26.5, 48.0)])

    ax.text(23.0, 72.8,
            "Dual-Safe Corridor (eta_rec >= 81%, 48/576 cells):\n100.0% Viability & Mineral <= 1.0x (min = 0.976x)",
            ha="center", va="center", fontsize=6.7, fontweight="bold", color="#0B4F3A",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", alpha=0.92, ec="#0B4F3A", lw=0.8),
            zorder=6)
    ax.set_xlabel("Social & Eco-Transition Transfer Budget Cap $g_{trans,\\max}$ (% of GDP)")
    ax.set_ylabel("Critical Mineral Recycling Efficiency $\\eta_{rec}$ (%)")
    ax.set_title("China 24×24 Circular-Welfare Policy Frontier (Viability = 100%)")

    out = FIG_DIR / "Fig5_China_ZeroLeakage_5Way_Factorial.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ms_t2, ms_t1, m5_cal, ds = load_all()
    make_fig1(ms_t2, ds)
    make_fig2(ms_t2, ms_t1)
    make_fig3(ms_t2)
    make_fig4(ms_t2, m5_cal)
    make_fig5(ms_t2, ds)
    print("[ALL 5 FIGURES GENERATED AT 600 DPI]")


if __name__ == "__main__":
    main()

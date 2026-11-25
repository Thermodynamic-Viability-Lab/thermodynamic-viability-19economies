"""
Shared model definitions, parameter priors, country-level data loader, and EXACT CPU reference mirror
of `sfc_bp_open_economy.cu` for:
  1. GPU-vs-CPU parity verification (< 1e-11 max difference across identical Philox-4x32-10 streams).
  2. Standalone 15-second referee CPU accounting-closure verification (`verify_cpu_closure_fast.py`).
"""
import math
import os
import numpy as np
import pandas as pd

PARAM_NAMES = [
    "rho_c", "c_max", "gamma_C", "gamma_E", "gamma_D", "gamma_O",
    "kappa_org", "delta_ent", "psi_trade", "gamma_trade",
    "epbt_base", "lambda_learn", "eta_rec",
    "u_crit", "def_crit", "theta_rev", "pi_crit", "tau_max",
    "k_m_infra", "k_t_age", "k_cb_strain", "shock_scale",
]

# Calibrated baseline matching validated Open-Economy SFC-BP parameterization
DEFAULT_PARAMS = np.array([
    0.025, 1.080, 0.125, 1.050, 0.450, 0.058,
    2.750, 0.082, 0.025, 0.250,
    2.200, 1.000, 0.450,
    0.115, 0.042, 0.720, 0.120, 0.850,
    1.150, 0.480, 3.200, 1.000,
], dtype=np.float64)

# 16 identifiable historical parameters calibrated in Mod 2 (ABC-SMC).
# Thresholds (u_crit, def_crit, theta_rev, pi_crit), lambda_learn=1.0, and eta_rec=0.45 are held fixed
# during historical 18-OECD calibration to prevent threshold softening or confounding.
ABC_ACTIVE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21]

ABC_PRIOR_BOUNDS = np.array([
    [0.014, 0.040],  # 0: rho_c
    [0.750, 1.450],  # 1: c_max
    [0.065, 0.195],  # 2: gamma_C
    [0.550, 1.650],  # 3: gamma_E
    [0.250, 0.680],  # 4: gamma_D
    [0.032, 0.090],  # 5: gamma_O
    [1.600, 4.000],  # 6: kappa_org
    [0.045, 0.130],  # 7: delta_ent
    [0.008, 0.055],  # 8: psi_trade
    [0.060, 0.550],  # 9: gamma_trade
    [1.200, 3.600],  # 10: epbt_base
    [0.680, 0.960],  # 17: tau_max
    [0.650, 1.800],  # 18: k_m_infra
    [0.280, 0.750],  # 19: k_t_age
    [1.600, 5.000],  # 20: k_cb_strain
    [0.650, 1.450],  # 21: shock_scale
], dtype=np.float64)

# Full 22-parameter ranges for Mod 3 Saltelli-Sobol Global Variance Decomposition
SOBOL_BOUNDS = np.array([
    [0.016, 0.036],  # 0: rho_c
    [0.800, 1.380],  # 1: c_max
    [0.080, 0.180],  # 2: gamma_C
    [0.650, 1.500],  # 3: gamma_E
    [0.300, 0.620],  # 4: gamma_D
    [0.038, 0.082],  # 5: gamma_O
    [1.800, 3.800],  # 6: kappa_org
    [0.052, 0.118],  # 7: delta_ent
    [0.010, 0.048],  # 8: psi_trade
    [0.080, 0.480],  # 9: gamma_trade
    [1.400, 3.400],  # 10: epbt_base
    [0.600, 1.250],  # 11: lambda_learn
    [0.200, 0.850],  # 12: eta_rec
    [0.088, 0.145],  # 13: u_crit
    [0.032, 0.056],  # 14: def_crit
    [0.560, 0.900],  # 15: theta_rev
    [0.090, 0.155],  # 16: pi_crit
    [0.700, 0.950],  # 17: tau_max
    [0.750, 1.650],  # 18: k_m_infra
    [0.320, 0.700],  # 19: k_t_age
    [2.000, 4.600],  # 20: k_cb_strain
    [0.700, 1.350],  # 21: shock_scale
], dtype=np.float64)

# Regime array layout (11 float64 values matching CUDA kernel Reg[11]):
# [zero_growth, r_0, okun_g, hard_money, g_maint_max, g_trans_max, g_org_min, g_org_max, mu_cb_max, coercive_mode, taper_years]
REGIMES = {
    "1_Adaptive_Growth": np.array([0, 0.022, 0.012, 0, 0.36, 0.32, 0.020, 0.085, 0.85, 0, 30.0], dtype=np.float64),
    "2_Austerity_Coercion": np.array([0, 0.022, 0.012, 0, 0.135, 0.065, 0.115, 0.165, 0.65, 1, 30.0], dtype=np.float64),
    "3_Neoliberal_Cap": np.array([0, 0.022, 0.012, 0, 0.145, 0.090, 0.020, 0.050, 0.80, 0, 30.0], dtype=np.float64),
    "4_Gold_Standard": np.array([0, 0.028, 0.012, 1, 0.085, 0.040, 0.020, 0.035, 0.00, 0, 30.0], dtype=np.float64),
    "5A_PG_Unreformed": np.array([1, 0.028, 0.012, 1, 0.36, 0.32, 0.020, 0.070, 0.00, 0, 30.0], dtype=np.float64),
    "5M_PG_Monetary_Only": np.array([1, 0.002, 0.012, 0, 0.36, 0.32, 0.020, 0.070, 0.92, 0, 30.0], dtype=np.float64),
    "5W_PG_WorkShare_Only": np.array([1, 0.028, 0.000, 1, 0.36, 0.32, 0.020, 0.070, 0.00, 0, 30.0], dtype=np.float64),
    "5B_PG_EcoSocial_Full": np.array([1, 0.002, 0.000, 0, 0.36, 0.32, 0.020, 0.070, 0.92, 0, 30.0], dtype=np.float64),
    # Forward post-2024 switch regimes (taper_years = 10.0)
    "FWD_PostGrowth_EcoSocial": np.array([1, 0.002, 0.000, 0, 0.36, 0.32, 0.020, 0.070, 0.92, 0, 10.0], dtype=np.float64),
    # China Symmetric 5-Way Controlled Factorial Pathways (shared OECD posterior, zero China tuning)
    "CHN_Hist_Base": np.array([0, 0.012, 0.010, 0, 0.30, 0.16, 0.035, 0.078, 0.85, 0, 30.0], dtype=np.float64),
    "CHN_A_Austerity_SlowElec": np.array([0, 0.024, 0.012, 1, 0.135, 0.080, 0.025, 0.055, 0.00, 0, 10.0], dtype=np.float64),
    "CHN_B_FastElec_Under_Austerity": np.array([0, 0.024, 0.012, 1, 0.135, 0.080, 0.025, 0.055, 0.00, 0, 10.0], dtype=np.float64),
    "CHN_C_Sovereign_FossilInertia": np.array([0, 0.004, 0.010, 0, 0.300, 0.155, 0.040, 0.072, 0.90, 0, 10.0], dtype=np.float64),
    "CHN_D_Current_Elec_Infra_Coord": np.array([0, 0.004, 0.010, 0, 0.300, 0.155, 0.040, 0.072, 0.90, 0, 10.0], dtype=np.float64),
    "CHN_E_Elec_Plus_EcoSocial_Rebalance": np.array([1, 0.002, 0.002, 0, 0.340, 0.290, 0.035, 0.068, 0.92, 0, 10.0], dtype=np.float64),
}


def load_experiment_datasets(data_dir):
    """Loads pre-built empirical 18-OECD panel, UN WPP 19-country aging/trade table, and China OOS panel."""
    oecd = pd.read_csv(os.path.join(data_dir, "Clean_18OECD_Panel_1920_2024.csv"))
    age19 = pd.read_csv(os.path.join(data_dir, "UN_WPP_19Country_AgeDep_1920_2050.csv"))
    chn = pd.read_csv(os.path.join(data_dir, "Clean_China_OOS_Panel_1950_2024.csv"))

    isos = sorted(oecd["iso"].unique().tolist())
    n_c = len(isos)
    T_hist = 105
    assert n_c == 18 and len(oecd) == n_c * T_hist

    shape_h = (n_c, T_hist)
    arrays = {k: np.zeros(shape_h, dtype=np.float64) for k in [
        "eroi", "eta", "age", "civ", "nonf", "net_trade",
        "obs_g", "obs_debt", "obs_pi", "obs_u"
    ]}
    arrays["is_shock"] = np.zeros(shape_h, dtype=np.int32)
    arrays["is_imp"] = np.zeros(n_c, dtype=np.int32)
    arrays["init_pub_debt"] = np.zeros(n_c, dtype=np.float64)
    arrays["init_priv_debt"] = np.zeros(n_c, dtype=np.float64)
    arrays["etwh_2024"] = np.zeros(n_c, dtype=np.float64)
    arrays["co2_2024_gt"] = np.zeros(n_c, dtype=np.float64)
    arrays["clim_b15"] = np.zeros(n_c, dtype=np.float64)
    arrays["min_caps"] = np.zeros((n_c, 3), dtype=np.float64)
    arrays["age_131"] = np.zeros((n_c, 131), dtype=np.float64)
    arrays["net_trade_131"] = np.zeros((n_c, 131), dtype=np.float64)

    oecd_2024 = oecd[oecd["year"] == 2024].set_index("iso")
    tot_co2_2024 = float(oecd_2024["co2_mt"].sum() / 1000.0)
    tot_etwh_2024 = float(oecd_2024["primary_energy_twh"].sum())

    for c_i, iso in enumerate(isos):
        sub = oecd[oecd["iso"] == iso].sort_values("year")
        sub_a = age19[age19["iso"] == iso].sort_values("year")
        arrays["eroi"][c_i] = sub["composite_primary_eroi"].values
        arrays["eta"][c_i] = sub["exergy_efficiency_eta"].values
        arrays["age"][c_i] = sub["old_age_dep_ratio"].values
        arrays["civ"][c_i] = 0.40 + 0.80 * sub["vdem_civsoc_idx"].values
        arrays["nonf"][c_i] = sub["non_fossil_share"].values
        arrays["net_trade"][c_i] = sub_a[sub_a["year"] <= 2024]["net_energy_trade_share"].values
        arrays["obs_g"][c_i] = sub["g_gdp"].values
        arrays["obs_debt"][c_i] = sub["total_debt_gdp"].values
        arrays["obs_pi"][c_i] = sub["inflation_cpi"].values
        arrays["obs_u"][c_i] = sub["unemp"].values
        arrays["is_shock"][c_i] = sub["ww2_pandemic_shock"].values.astype(np.int32)
        arrays["is_imp"][c_i] = int(sub["is_importer"].iloc[0])
        arrays["init_pub_debt"][c_i] = float(sub["public_debt_gdp"].iloc[0])
        arrays["init_priv_debt"][c_i] = float(sub["priv_credit_gdp"].iloc[0])
        e24 = float(sub["primary_energy_twh"].iloc[-1])
        c24 = float(sub["co2_mt"].iloc[-1] / 1000.0)
        arrays["etwh_2024"][c_i] = e24
        arrays["co2_2024_gt"][c_i] = c24
        arrays["clim_b15"][c_i] = 85.0 * (c24 / tot_co2_2024)
        e_share = e24 / tot_etwh_2024
        arrays["min_caps"][c_i] = np.array([115.0 * e_share, 14.0 * e_share, 1500.0 * e_share], dtype=np.float64)
        arrays["age_131"][c_i] = sub_a["old_age_dep_ratio"].values
        arrays["net_trade_131"][c_i] = sub_a["net_energy_trade_share"].values

    chn = chn.sort_values("year").reset_index(drop=True)
    chn_a = age19[(age19["iso"] == "CHN") & (age19["year"] >= 1950)].sort_values("year")
    assert len(chn) == 75 and len(chn_a) == 101
    chn_e24 = float(chn["primary_energy_twh"].iloc[-1])
    chn_c24 = float(chn["co2_mt"].iloc[-1] / 1000.0)
    chn_dict = {
        "years": chn_a["year"].values.astype(np.int32),
        "eroi_hist": chn["composite_primary_eroi"].values.astype(np.float64),
        "eta_hist": chn["exergy_efficiency_eta"].values.astype(np.float64),
        "age_hist": chn["old_age_dep_ratio"].values.astype(np.float64),
        "civ_hist": (0.40 + 0.80 * chn["vdem_civsoc_idx"].values).astype(np.float64),
        "nonf_hist": chn["non_fossil_share"].values.astype(np.float64),
        "net_trade_hist": chn["net_energy_trade_share"].values.astype(np.float64),
        "age_101": chn_a["old_age_dep_ratio"].values.astype(np.float64),
        "net_trade_101": chn_a["net_energy_trade_share"].values.astype(np.float64),
        "obs_g": chn["obs_g_gdp"].values.astype(np.float64),
        "obs_debt": chn["obs_public_debt_gdp"].values.astype(np.float64),
        "obs_rgdp": chn["obs_real_gdp_growth"].values.astype(np.float64),
        "etwh_2024": chn_e24,
        "co2_2024_gt": chn_c24,
        "clim_budget_chn": 140.0,
        "min_caps": np.array([105.0, 12.8, 1380.0], dtype=np.float64),
        "init_pub_debt": 0.25,
        "init_priv_debt": 0.35,
    }

    return {"isos": isos, "oecd": arrays, "china": chn_dict}


# -----------------------------------------------------------------------------
# Bit-exact Python mirror of CUDA Philox-4x32-10 PRNG & SFC-BP step
# -----------------------------------------------------------------------------
def _philox_round_py(c0, c1, c2, c3, k0, k1):
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    p0 = M0 * c0
    p1 = M1 * c2
    hi0 = (p0 >> 32) & 0xFFFFFFFF
    lo0 = p0 & 0xFFFFFFFF
    hi1 = (p1 >> 32) & 0xFFFFFFFF
    lo1 = p1 & 0xFFFFFFFF
    nc0 = (hi1 ^ c1 ^ k0) & 0xFFFFFFFF
    nc1 = lo1
    nc2 = (hi0 ^ c3 ^ k1) & 0xFFFFFFFF
    nc3 = lo0
    return nc0, nc1, nc2, nc3


def philox4x32_10_normals_py(seed, stream_id, step_t, sub_id):
    c0 = int(stream_id) & 0xFFFFFFFF
    c1 = int(step_t) & 0xFFFFFFFF
    c2 = int(sub_id) & 0xFFFFFFFF
    c3 = 0x20261122
    k0 = int(seed) & 0xFFFFFFFF
    k1 = (int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85

    for _ in range(10):
        c0, c1, c2, c3 = _philox_round_py(c0, c1, c2, c3, k0, k1)
        k0 = (k0 + W0) & 0xFFFFFFFF
        k1 = (k1 + W1) & 0xFFFFFFFF

    inv2_32 = 2.3283064365386963e-10
    u0 = (float(c0) + 0.5) * inv2_32
    u1 = (float(c1) + 0.5) * inv2_32
    u2 = (float(c2) + 0.5) * inv2_32
    r_rad = math.sqrt(-2.0 * math.log(u0))
    theta = 2.0 * math.pi * u1
    return r_rad * math.cos(theta), r_rad * math.sin(theta), u2


def clampd_py(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def init_country_state_py(coercive_mode, init_pub_debt, init_priv_debt):
    S = np.zeros(25, dtype=np.float64)
    d_pub = clampd_py(init_pub_debt, 0.15, 1.10)
    d_priv = clampd_py(init_priv_debt, 0.15, 0.85)
    S[0] = 1.0
    S[1] = 100.0
    S[2] = 220.0
    S[3] = 75.0
    S[4] = 100.0 * d_priv
    S[5] = 100.0 * d_pub * 0.80
    S[6] = 100.0 * d_pub * 0.20
    S[8] = 0.0
    S[7] = S[4] + S[6] + S[8]
    S[9] = 0.25
    S[10] = 0.0
    S[11] = 0.015
    S[12] = 0.015
    S[13] = 0.048
    S[14] = 0.008
    S[15] = 0.56
    S[16] = 0.14
    S[17] = 0.095
    S[18] = 0.032
    S[19] = 0.115 if coercive_mode > 0.5 else 0.048
    S[20] = 0.0
    return S


def step_country_sfc_bp_py(
    S, Prm, Reg, t, t_switch, t_build_start,
    eroi_t, eta_t, age_t, civ_t, nonf_t, nonf_prev,
    impdep_signed_t, co2_mult_t, c_mult_t,
    e_twh_2024, co2_2024_gt, clim_budget_15, min_caps,
    enable_climate_fb, is_post2024_switch,
    eps_real, eps_inf, u_haz, state_scalars,
    blind_no_age=0, trade_absorb_frac=0.75, linear_clim_mode=0
):
    P, Y_real, K_p, K_g = S[0], S[1], S[2], S[3]
    L_f, B_h, B_cb, M_h, FX = S[4], S[5], S[6], S[7], S[8]
    B = B_h + B_cb
    Y_nom = P * Y_real

    C, B_infra, pi_e, pi = S[9], S[10], S[11], S[12]
    u, def_rate, O_state, O_rev = S[13], S[14], S[15], S[16]
    g_maint, g_trans, g_org, mu_cb = S[17], S[18], S[19], S[20]

    (rho_c, c_max, gamma_C, gamma_E, gamma_D, gamma_O,
     kappa_org, delta_ent, psi_trade, gamma_trade,
     epbt_base, lambda_learn, eta_rec,
     u_crit, def_crit, theta_rev, pi_crit, tau_max,
     k_m_infra, k_t_age, k_cb_strain, shock_scale) = [float(x) for x in Prm]

    fat_shock = clampd_py(eps_real * (1.0 + 0.06 * eps_real * eps_real), -3.2, 3.2) * shock_scale
    eroi_safe = max(eroi_t, 1.05)
    eta_safe = max(eta_t, 0.08)
    inv_eroi = 1.0 / eroi_safe
    nf_raw = max(0.10, 1.0 - inv_eroi)

    r_cu = S[22] / max(float(min_caps[0]), 1e-6)
    r_li = S[23] / max(float(min_caps[1]), 1e-6)
    r_st = S[24] / max(float(min_caps[2]), 1e-6)
    min_overshoot = max(0.0, max(r_cu, r_li, r_st) - 1.0)

    d_nonf = clampd_py(nonf_t - nonf_prev, 0.0, 0.045)
    vre_repower = (max(0.0, nonf_t - 0.16) / 35.0) if t >= t_switch else 0.0
    build_rate = min(0.065, d_nonf + vre_repower) if t >= t_build_start else 0.0
    epbt_eff = epbt_base * lambda_learn * (1.0 + 0.35 * min_overshoot)
    build_frac = min(0.22, epbt_eff * build_rate) if t >= t_build_start else 0.0
    build_frac = min(build_frac, max(0.0, 0.90 - inv_eroi))

    inv_eroi_dyn = inv_eroi + build_frac
    nf_dyn = max(0.10, 1.0 - inv_eroi_dyn)

    E_gross = Y_real * (0.24 / eta_safe)
    E_req = E_gross * inv_eroi
    E_build = E_gross * build_frac
    E_net = E_gross - E_req - E_build
    Q_waste = (1.0 - eta_safe) * E_net
    B_net = eta_safe * E_net
    phys_err = abs(E_gross - (E_req + E_build + Q_waste + B_net)) / max(E_gross, 1e-9)
    if phys_err > state_scalars["max_phys_err"]:
        state_scalars["max_phys_err"] = phys_err

    if t == t_switch - 1 and state_scalars["e_gross_ref"] <= 0.0:
        state_scalars["e_gross_ref"] = E_gross

    if t >= t_switch:
        ref_e = state_scalars["e_gross_ref"] if state_scalars["e_gross_ref"] > 1.0 else E_gross
        co2_rate = co2_2024_gt * (E_gross / ref_e) * co2_mult_t
        S[21] += max(0.0, co2_rate)
        e_twh_t = e_twh_2024 * (E_gross / ref_e)
        added_twh = e_twh_t * build_rate
        repl_share = vre_repower / max(build_rate, 1e-6)
        rec_factor = 1.0 - eta_rec * clampd_py(repl_share, 0.0, 1.0)
        S[22] += added_twh * 0.00420 * lambda_learn * rec_factor
        S[23] += added_twh * 0.00052 * lambda_learn * rec_factor
        S[24] += added_twh * 0.05200 * lambda_learn * rec_factor

    clim_wear = 0.0
    if enable_climate_fb and t >= t_switch:
        ov15 = max(0.0, (S[21] - clim_budget_15) / max(clim_budget_15, 5.0))
        clim_wear = (0.042 * ov15) if linear_clim_mode else (0.032 * ov15 + 0.022 * ov15 * ov15)

    impdep_eff = impdep_signed_t * clampd_py((1.0 - nonf_t) / 0.84, 0.05, 1.20)
    nx_energy = clampd_py(-psi_trade * impdep_eff * (inv_eroi_dyn * 15.0), -0.10, 0.10)
    fx_ratio_prev = FX / max(Y_nom, 1e-6)
    nx_other = -trade_absorb_frac * nx_energy - 0.14 * fx_ratio_prev
    nx_tot = clampd_py(nx_energy + nx_other, -0.08, 0.08)
    import_drain = max(0.0, -nx_energy)
    export_buffer = max(0.0, nx_energy)

    c_cap = max(c_max * nf_raw, 0.25)
    rho_eff = rho_c * (c_mult_t if t >= t_switch else 1.0)
    C = clampd_py(C + rho_eff * C * (1.0 - C / c_cap), 0.05, 2.5)
    c_over_nf = C / nf_dyn

    g_maint_req = max(
        0.02,
        gamma_C * c_over_nf + gamma_E * inv_eroi_dyn
        + gamma_trade * import_drain - 0.30 * gamma_trade * export_buffer + clim_wear
    )
    g_trans_req = gamma_D * age_t

    rratio_prev = O_rev / max(O_state, 0.08)
    d_prev = (B + L_f) / max(Y_nom, 1e-6)
    u_gap = max(0.0, u - 0.050)
    r_gap = max(0.0, rratio_prev - 0.36)
    pi_excess = max(0.0, pi - 0.050)
    o_gap = max(0.0, 0.72 - O_state)

    k_m_ease, k_t_unemp, k_o_cap, k_pi_brake = 0.16, 0.45, 0.22, 0.30
    dg_m = 0.35 * (
        k_m_infra * max(0.0, B_infra - 0.004)
        - k_m_ease * ((pi_excess + 0.008) if B_infra < 0.003 else 0.0)
        - k_pi_brake * pi_excess
    )
    if blind_no_age:
        u_excess = max(0.0, u - 0.068)
        u_calm = max(0.0, 0.064 - u)
        dg_t = 0.45 * (k_t_unemp * u_excess + k_t_age * r_gap - k_m_ease * u_calm - k_pi_brake * pi_excess)
    else:
        g_trans_stat = k_t_age * age_t + k_t_unemp * u_gap + 0.08 * r_gap
        dg_t = 0.32 * (g_trans_stat - g_trans) - 0.25 * k_pi_brake * pi_excess
    if Reg[9] > 0.5:
        dg_o = 0.18 * (0.045 + 0.12 * r_gap)
    else:
        g_org_stat = gamma_O * 0.74 + k_o_cap * o_gap + 0.05 * r_gap
        dg_o = 0.28 * (g_org_stat - g_org)

    g_maint = clampd_py(g_maint + clampd_py(dg_m, -0.015, 0.024), 0.040, clampd_py(Reg[4], 0.05, 0.45))
    g_trans = clampd_py(g_trans + clampd_py(dg_t, -0.015, 0.022), 0.018, clampd_py(Reg[5], 0.03, 0.40))
    g_org = clampd_py(g_org + clampd_py(dg_o, -0.008, 0.014), Reg[6], clampd_py(Reg[7], Reg[6], 0.22))
    g_tot = g_maint + g_trans + g_org

    if Reg[3] > 0.5:
        mu_cb = 0.0
    else:
        mu_target = clampd_py(
            0.15 + k_cb_strain * max(0.0, def_rate - 0.011) + 0.20 * max(0.0, d_prev - 1.30) - 2.0 * pi_excess,
            0.0, Reg[8]
        )
        mu_cb = 0.65 * mu_cb + 0.35 * mu_target

    maint_short = max(0.0, g_maint_req - g_maint)
    trans_short = max(0.0, g_trans_req - g_trans)
    B_infra = clampd_py(0.55 * B_infra + maint_short + 0.003 * abs(fat_shock), 0.0, 0.40)
    D_crowd = 0.65 * max(0.0, g_org - 0.075)

    if Reg[0] > 0.5:
        phi_cap = (max(0.0, 1.0 - float(t - t_switch) / max(Reg[10], 1.0))
                   if (is_post2024_switch and t >= t_switch)
                   else max(0.0, 1.0 - float(t) / max(Reg[10], 1.0)))
        g0 = 0.020 if (is_post2024_switch and t >= t_switch) else 0.016
        g_Y_pot = g0 * phi_cap * (nf_dyn / 0.92)
    else:
        g_Y_pot = 0.023 * nf_dyn * ((eta_safe / 0.17) ** 0.45)

    tau_cap = tau_max * O_state
    fiscal_overheat = max(0.0, g_tot - (tau_cap + 0.04))
    g_Y = clampd_py(
        g_Y_pot - 0.22 * B_infra - 0.28 * D_crowd - 0.35 * fiscal_overheat + 0.0065 * fat_shock,
        -0.065, 0.065
    )
    Y_real_new = max(1.0, Y_real * (1.0 + g_Y))

    i_b = max(0.002, Reg[1] + 0.85 * pi_e - 0.018 * mu_cb)
    i_l = i_b + 0.018 + 0.25 * def_rate
    i_fx = 0.020

    pi = clampd_py(
        0.42 * pi_e + 0.022 * mu_cb + 0.18 * inv_eroi_dyn + 0.85 * fiscal_overheat
        + 0.18 * import_drain - 0.10 * (maint_short + trans_short) + 0.007 * eps_inf * shock_scale,
        -0.045, 0.30
    )
    pi_e = pi_e + 0.35 * (pi - pi_e)
    P_new = max(0.05, P * (1.0 + pi))
    Y_nom_new = P_new * Y_real_new

    NX_E = nx_tot * Y_nom_new
    G_maint_nom = g_maint * Y_nom_new
    G_org_nom = g_org * Y_nom_new
    T_trans_nom = g_trans * Y_nom_new

    tau_eff = min(tau_cap, 0.94 * g_tot + 0.015 * max(0.0, B / max(Y_nom_new, 1e-6) - 0.65))
    T_tax_nom = tau_eff * Y_nom_new

    Int_bh = i_b * B_h
    Int_lf = i_l * L_f
    Int_fx = i_fx * FX

    dB = G_maint_nom + G_org_nom + T_trans_nom + Int_bh - T_tax_nom
    dB_cb = (mu_cb * dB) if dB > 0.0 else 0.0
    if Reg[3] <= 0.5:
        qe_add = mu_cb * max(0.0, 0.12 * (B_h - 0.75 * Y_nom_new))
        dB_cb = min(dB_cb + qe_add, max(0.0, B_h + max(dB, 0.0) * 0.98))
    dB_h = dB - dB_cb

    dep_p = 0.045
    K_p_target = 2.20 * Y_real_new
    I_p_real_raw = max(0.02 * Y_real_new, dep_p * K_p + 0.25 * (K_p_target - K_p))
    I_p_nom_raw = P_new * I_p_real_raw
    I_p_room = max(0.01 * Y_nom_new, 0.85 * Y_nom_new - G_maint_nom - G_org_nom - NX_E)
    I_p_nom = min(I_p_nom_raw, I_p_room)
    I_p_real = I_p_nom / P_new
    C_p_nom = Y_nom_new - I_p_nom - G_maint_nom - G_org_nom - NX_E

    w_org_wage = 0.65
    firm_gross_rev = C_p_nom + I_p_nom + G_maint_nom + (1.0 - w_org_wage) * G_org_nom + NX_E
    FU_f = clampd_py(
        0.14 * firm_gross_rev - 0.35 * Int_lf - 0.20 * (B_infra + D_crowd) * Y_nom_new,
        -0.08 * Y_nom_new, 0.25 * Y_nom_new
    )
    Y_f_nom = firm_gross_rev - Int_lf - FU_f
    dL_f = I_p_nom - FU_f

    YD_h = Y_f_nom + w_org_wage * G_org_nom + T_trans_nom - T_tax_nom + Int_bh + Int_lf + Int_fx
    Sav_h = YD_h - C_p_nom
    dFX = NX_E + Int_fx
    dM_h = Sav_h - dB_h

    B_h_new = B_h + dB_h
    B_cb_new = B_cb + dB_cb
    B_new = B_h_new + B_cb_new
    L_f_new = L_f + dL_f
    FX_new = FX + dFX
    M_h_new = M_h + dM_h

    K_p_new = max(1.0, K_p * (1.0 - dep_p) + I_p_real)
    delta_kg = 0.04 + 0.08 * c_over_nf + clim_wear
    K_g_new = max(1.0, K_g * (1.0 - delta_kg) + g_maint * Y_real_new)

    mon_err = abs(M_h_new - (L_f_new + B_cb_new + FX_new)) / max(Y_nom_new, 1e-6)
    if mon_err > state_scalars["max_mon_err"]:
        state_scalars["max_mon_err"] = mon_err

    NW_h = M_h_new + B_h_new
    NW_f = P_new * K_p_new - L_f_new
    NW_g = P_new * K_g_new - B_new
    NW_b = (L_f_new + B_cb_new + FX_new) - M_h_new
    NW_row = -FX_new
    nw_err = abs((NW_h + NW_f + NW_g + NW_b + NW_row) - P_new * (K_p_new + K_g_new)) / max(Y_nom_new, 1e-6)
    if nw_err > state_scalars["max_nw_err"]:
        state_scalars["max_nw_err"] = nw_err

    d_priv = max(0.0, L_f_new) / max(Y_nom_new, 1e-6)
    d_pub_mkt = max(0.0, B_h_new) / max(Y_nom_new, 1e-6)
    d_tot = (max(0.0, L_f_new) + max(0.0, B_new)) / max(Y_nom_new, 1e-6)
    bop_strain = max(0.0, -FX_new / max(Y_nom_new, 1e-6) - 0.25)

    priv_surplus = max(0.10, (1.0 - g_maint - g_org) * nf_dyn)
    r_real_mkt = max(0.0, i_l - pi)
    r_real_pub = i_b - pi
    soddy_gap = max(0.0, r_real_pub - g_Y) * d_pub_mkt + max(0.0, (r_real_mkt - g_Y) - 0.018) * d_priv
    dsr = (i_l * max(0.0, L_f_new) + i_b * max(0.0, B_h_new)) / max(Y_nom_new, 1e-6)
    solv_strain = clampd_py(
        1.35 * soddy_gap + max(0.0, dsr - 0.16 * priv_surplus) * 1.10 + 0.25 * bop_strain - 0.032 * mu_cb,
        0.0, 0.35
    )

    okun_gap = max(0.0, Reg[2] - g_Y)
    u_new = clampd_py(
        0.35 * u + 0.65 * (0.044 + 1.25 * B_infra + 0.95 * (maint_short + trans_short)
                           + 1.05 * D_crowd + 0.60 * solv_strain + 2.65 * okun_gap + 0.011 * abs(fat_shock)),
        0.015, 0.35
    )
    def_new = clampd_py(
        0.30 * def_rate + 0.70 * (0.008 + 1.40 * solv_strain + 0.45 * B_infra + 0.90 * okun_gap + 0.006 * abs(fat_shock)),
        0.002, 0.25
    )
    O_state_new = clampd_py(
        O_state + kappa_org * g_org * (1.0 - O_state)
        - (delta_ent * c_over_nf + 1.65 * solv_strain + 1.30 * B_infra) * O_state,
        0.08, 0.98
    )
    grievance = (
        1.90 * max(0.0, u_new - 0.048) + 1.65 * trans_short
        + 1.10 * solv_strain + 0.80 * max(0.0, pi - 0.045)
    )
    O_rev_new = clampd_py(
        O_rev + 1.85 * grievance * civ_t * O_rev * (1.0 - O_rev)
        - (2.15 * O_state_new * g_org + 1.35 * g_trans) * O_rev,
        0.02, 0.98
    )
    rratio = O_rev_new / O_state_new

    ex_l = max(0.0, rratio - theta_rev)
    ex_u = max(0.0, u_new - u_crit)
    ex_d = max(0.0, def_new - def_crit)
    ex_p = max(0.0, pi - pi_crit)
    haz = 3.20 * ex_l + 2.40 * ex_u + 2.95 * ex_d + 2.50 * ex_p
    p_fail = 0.26 * haz

    if state_scalars["alive"] and (haz > 1e-4) and (u_haz < p_fail):
        state_scalars["alive"] = 0
        state_scalars["cause"] = 1 if ex_l > 0.0 else 2
    if state_scalars["alive"]:
        state_scalars["years_alive"] += 1.0

    S[0], S[1], S[2], S[3] = P_new, Y_real_new, K_p_new, K_g_new
    S[4], S[5], S[6], S[7], S[8] = L_f_new, B_h_new, B_cb_new, M_h_new, FX_new
    S[9], S[10], S[11], S[12], S[13] = C, B_infra, pi_e, pi, u_new
    S[14], S[15], S[16] = def_new, O_state_new, O_rev_new
    S[17], S[18], S[19], S[20] = g_maint, g_trans, g_org, mu_cb

    return NX_E, g_tot, d_tot, rratio, g_Y

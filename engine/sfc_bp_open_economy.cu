// =============================================================================
// Closed 4-Sector + Rest-of-World Stock-Flow-Fund Biophysical Engine (FP64)
// Forward-compatible CUDA C++ (NVRTC compute_80 -> native sm_89 / sm_120 JIT)
// Zero external #include headers; zero neural-network black-box weights.
// Enforces at every trajectory-year:
//   1. Monetary Redundant Identity:  |M_h - (L_f + B_cb + FX)| / Y_nom < 1e-12
//   2. 5-Sector Net Worth Identity:  |sum(NW) - P*(K_p + K_g)| / Y_nom < 1e-12
//   3. 1st & 2nd Law + Build Energy: |E_gross - (E_req + E_build + Q_waste + B_net)| / E_gross < 1e-12
// =============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Deterministic Philox-4x32-10 PRNG (exact match with cpu_reference.py)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void philox_round(
    unsigned int* c0, unsigned int* c1, unsigned int* c2, unsigned int* c3,
    unsigned int k0, unsigned int k1
) {
    const unsigned int M0 = 0xD2511F53U;
    const unsigned int M1 = 0xCD9E8D57U;
    unsigned long long p0 = (unsigned long long)M0 * (unsigned long long)(*c0);
    unsigned long long p1 = (unsigned long long)M1 * (unsigned long long)(*c2);
    unsigned int hi0 = (unsigned int)(p0 >> 32);
    unsigned int lo0 = (unsigned int)(p0 & 0xFFFFFFFFULL);
    unsigned int hi1 = (unsigned int)(p1 >> 32);
    unsigned int lo1 = (unsigned int)(p1 & 0xFFFFFFFFULL);

    unsigned int nc0 = hi1 ^ (*c1) ^ k0;
    unsigned int nc1 = lo1;
    unsigned int nc2 = hi0 ^ (*c3) ^ k1;
    unsigned int nc3 = lo0;
    *c0 = nc0; *c1 = nc1; *c2 = nc2; *c3 = nc3;
}

__device__ __forceinline__ void philox4x32_10_normals(
    unsigned int seed, unsigned int stream_id, unsigned int step_t, unsigned int sub_id,
    double* eps_real, double* eps_inf, double* u_haz
) {
    unsigned int c0 = stream_id;
    unsigned int c1 = step_t;
    unsigned int c2 = sub_id;
    unsigned int c3 = 0x20261122U;
    unsigned int k0 = seed;
    unsigned int k1 = seed ^ 0x9E3779B9U;

    const unsigned int W0 = 0x9E3779B9U;
    const unsigned int W1 = 0xBB67AE85U;

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        philox_round(&c0, &c1, &c2, &c3, k0, k1);
        k0 += W0;
        k1 += W1;
    }

    const double inv2_32 = 2.3283064365386963e-10; // 1.0 / 2^32
    double u0 = ((double)c0 + 0.5) * inv2_32;
    double u1 = ((double)c1 + 0.5) * inv2_32;
    double u2 = ((double)c2 + 0.5) * inv2_32;

    double r_rad = sqrt(-2.0 * log(u0));
    double theta = 2.0 * M_PI * u1;
    *eps_real = r_rad * cos(theta);
    *eps_inf  = r_rad * sin(theta);
    *u_haz    = u2;
}

__device__ __forceinline__ double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// -----------------------------------------------------------------------------
// Parameter Vector Layout (22 doubles):
//   0:rho_c, 1:c_max, 2:gamma_C, 3:gamma_E, 4:gamma_D, 5:gamma_O,
//   6:kappa_org, 7:delta_ent, 8:psi_trade, 9:gamma_trade,
//  10:epbt_base, 11:lambda_learn, 12:eta_rec,
//  13:u_crit, 14:def_crit, 15:theta_rev, 16:pi_crit, 17:tau_max,
//  18:k_m_infra, 19:k_t_age, 20:k_cb_strain, 21:shock_scale
//
// Regime Config Layout (11 doubles):
//   0:zero_growth(0/1), 1:r_0, 2:okun_g, 3:hard_money(0/1),
//   4:g_maint_max, 5:g_trans_max, 6:g_org_min, 7:g_org_max,
//   8:mu_cb_max, 9:coercive_mode(0/1), 10:taper_years
//
// Country Mutable State Layout (25 doubles):
//   0:P, 1:Y_real, 2:K_p, 3:K_g, 4:L_f, 5:B_h, 6:B_cb, 7:M_h, 8:FX,
//   9:C, 10:B_infra, 11:pi_e, 12:pi, 13:u, 14:def_rate, 15:O_state, 16:O_rev,
//  17:g_maint, 18:g_trans, 19:g_org, 20:mu_cb,
//  21:co2_cum, 22:cu_cum, 23:li_cum, 24:st_cum
// -----------------------------------------------------------------------------

__device__ __forceinline__ void init_country_state(
    double* S, double coercive_mode, double init_pub_debt, double init_priv_debt
) {
    double d_pub  = clampd(init_pub_debt,  0.15, 1.10);
    double d_priv = clampd(init_priv_debt, 0.15, 0.85);
    S[0] = 1.0;                 // P
    S[1] = 100.0;               // Y_real
    S[2] = 220.0;               // K_p
    S[3] = 75.0;                // K_g
    S[4] = 100.0 * d_priv;      // L_f
    S[5] = 100.0 * d_pub * 0.80;// B_h
    S[6] = 100.0 * d_pub * 0.20;// B_cb
    S[8] = 0.0;                 // FX
    S[7] = S[4] + S[6] + S[8];  // M_h == L_f + B_cb + FX
    S[9] = 0.25;                // C
    S[10] = 0.0;                // B_infra
    S[11] = 0.015;              // pi_e
    S[12] = 0.015;              // pi
    S[13] = 0.048;              // u
    S[14] = 0.008;              // def_rate
    S[15] = 0.56;               // O_state
    S[16] = 0.14;               // O_rev
    S[17] = 0.095;              // g_maint
    S[18] = 0.032;              // g_trans
    S[19] = (coercive_mode > 0.5) ? 0.115 : 0.048; // g_org
    S[20] = 0.0;                // mu_cb
    S[21] = 0.0;                // co2_cum
    S[22] = 0.0;                // cu_cum
    S[23] = 0.0;                // li_cum
    S[24] = 0.0;                // st_cum
}

__device__ void step_country_sfc_bp(
    double* S,
    const double* Prm,
    const double* Reg,
    int t,
    int t_switch,
    int t_build_start,
    double eroi_t,
    double eta_t,
    double age_t,
    double civ_t,
    double nonf_t,
    double nonf_prev,
    double impdep_signed_t,
    double co2_mult_t,
    double c_mult_t,
    double e_twh_2024,
    double co2_2024_gt,
    double clim_budget_15,
    double min_cap_cu,
    double min_cap_li,
    double min_cap_st,
    int enable_climate_fb,
    int is_post2024_switch,
    int blind_no_age,
    double trade_absorb_frac,
    int linear_clim_mode,
    double eps_real,
    double eps_inf,
    double u_haz,
    double* e_gross_ref,
    int* alive,
    int* cause,
    double* years_alive,
    double* out_nx_nom,
    double* out_g_tot,
    double* out_d_tot,
    double* out_rratio,
    double* out_gY,
    double* max_mon_err,
    double* max_nw_err,
    double* max_phys_err,
    int* any_nan
) {
    double P = S[0], Y_real = S[1], K_p = S[2], K_g = S[3];
    double L_f = S[4], B_h = S[5], B_cb = S[6], M_h = S[7], FX = S[8];
    double B = B_h + B_cb;
    double Y_nom = P * Y_real;

    double C = S[9], B_infra = S[10], pi_e = S[11], pi = S[12];
    double u = S[13], def_rate = S[14], O_state = S[15], O_rev = S[16];
    double g_maint = S[17], g_trans = S[18], g_org = S[19], mu_cb = S[20];

    double rho_c        = Prm[0];
    double c_max        = Prm[1];
    double gamma_C      = Prm[2];
    double gamma_E      = Prm[3];
    double gamma_D      = Prm[4];
    double gamma_O      = Prm[5];
    double kappa_org    = Prm[6];
    double delta_ent    = Prm[7];
    double psi_trade    = Prm[8];
    double gamma_trade  = Prm[9];
    double epbt_base    = Prm[10];
    double lambda_learn = Prm[11];
    double eta_rec      = Prm[12];
    double u_crit       = Prm[13];
    double def_crit     = Prm[14];
    double theta_rev    = Prm[15];
    double pi_crit      = Prm[16];
    double tau_max      = Prm[17];
    double k_m_infra    = Prm[18];
    double k_t_age      = Prm[19];
    double k_cb_strain  = Prm[20];
    double shock_scale  = Prm[21];

    double fat_shock = clampd(eps_real * (1.0 + 0.06 * eps_real * eps_real), -3.2, 3.2) * shock_scale;
    double eroi_safe = fmax(eroi_t, 1.05);
    double eta_safe  = fmax(eta_t, 0.08);
    double inv_eroi  = 1.0 / eroi_safe;
    double nf_raw    = fmax(0.10, 1.0 - inv_eroi);

    double r_cu = S[22] / fmax(min_cap_cu, 1e-6);
    double r_li = S[23] / fmax(min_cap_li, 1e-6);
    double r_st = S[24] / fmax(min_cap_st, 1e-6);
    double min_overshoot = fmax(0.0, fmax(r_cu, fmax(r_li, r_st)) - 1.0);

    // 1. Construction-phase dynamic energy cannibalization & exact 1st/2nd Law closure
    //    Net expansion d_nonf incurs upfront embodied energy investment beyond lifecycle EROI;
    //    only active from t >= t_build_start (1965) to avoid pre-1965 CO2 backcast jumps.
    double d_nonf = clampd(nonf_t - nonf_prev, 0.0, 0.045);
    double vre_repower = (t >= t_switch) ? (fmax(0.0, nonf_t - 0.16) / 35.0) : 0.0;
    double build_rate = (t >= t_build_start) ? fmin(0.065, d_nonf + vre_repower) : 0.0;
    double epbt_eff = epbt_base * lambda_learn * (1.0 + 0.35 * min_overshoot);
    double build_frac = (t >= t_build_start) ? fmin(0.22, epbt_eff * build_rate) : 0.0;
    build_frac = fmin(build_frac, fmax(0.0, 0.90 - inv_eroi));

    double inv_eroi_dyn = inv_eroi + build_frac;
    double nf_dyn = fmax(0.10, 1.0 - inv_eroi_dyn);

    double E_gross = Y_real * (0.24 / eta_safe);
    double E_req   = E_gross * inv_eroi;
    double E_build = E_gross * build_frac;
    double E_net   = E_gross - E_req - E_build;
    double Q_waste = (1.0 - eta_safe) * E_net;
    double B_net   = eta_safe * E_net;
    double phys_err = fabs(E_gross - (E_req + E_build + Q_waste + B_net)) / fmax(E_gross, 1e-9);
    if (phys_err > *max_phys_err) *max_phys_err = phys_err;

    if (t == t_switch - 1 && *e_gross_ref <= 0.0) {
        *e_gross_ref = E_gross;
    }

    if (t >= t_switch) {
        double ref_e = (*e_gross_ref > 1.0) ? *e_gross_ref : E_gross;
        double co2_rate = co2_2024_gt * (E_gross / ref_e) * co2_mult_t;
        S[21] += fmax(0.0, co2_rate);

        double e_twh_t = e_twh_2024 * (E_gross / ref_e);
        double added_twh = e_twh_t * build_rate;
        double repl_share = vre_repower / fmax(build_rate, 1e-6);
        double rec_factor = 1.0 - eta_rec * clampd(repl_share, 0.0, 1.0);
        S[22] += added_twh * 0.00420 * lambda_learn * rec_factor; // Cu Mt (primary TWh eq.)
        S[23] += added_twh * 0.00052 * lambda_learn * rec_factor; // Li Mt (primary TWh eq.)
        S[24] += added_twh * 0.05200 * lambda_learn * rec_factor; // Steel Mt (primary TWh eq.)
    }

    double clim_wear = 0.0;
    if (enable_climate_fb && t >= t_switch) {
        double ov15 = fmax(0.0, (S[21] - clim_budget_15) / fmax(clim_budget_15, 5.0));
        clim_wear = (linear_clim_mode != 0) ? (0.042 * ov15) : (0.032 * ov15 + 0.022 * ov15 * ov15);
    }

    // Open-economy energy Terms-of-Trade + Merchandise/NFA stabilizing adjustment
    double impdep_eff = impdep_signed_t * clampd((1.0 - nonf_t) / 0.84, 0.05, 1.20);
    double nx_energy = clampd(-psi_trade * impdep_eff * (inv_eroi_dyn * 15.0), -0.10, 0.10);
    double fx_ratio_prev = FX / fmax(Y_nom, 1e-6);
    double nx_other = -trade_absorb_frac * nx_energy - 0.14 * fx_ratio_prev;
    double nx_tot = clampd(nx_energy + nx_other, -0.08, 0.08);
    double import_drain = fmax(0.0, -nx_energy);
    double export_buffer = fmax(0.0, nx_energy);

    // 2. Network Complexity & Biophysical Fiscal Requirement
    double c_cap = fmax(c_max * nf_raw, 0.25);
    double rho_eff = rho_c * ((t >= t_switch) ? c_mult_t : 1.0);
    C = clampd(C + rho_eff * C * (1.0 - C / c_cap), 0.05, 2.5);
    double c_over_nf = C / nf_dyn;

    double g_maint_req = fmax(
        0.02,
        gamma_C * c_over_nf + gamma_E * inv_eroi_dyn
        + gamma_trade * import_drain - 0.30 * gamma_trade * export_buffer + clim_wear
    );
    double g_trans_req = gamma_D * age_t;

    // 3. Observable-Feedback Fiscal & Monetary Policy Rule
    double rratio_prev = O_rev / fmax(O_state, 0.08);
    double d_prev = (B + L_f) / fmax(Y_nom, 1e-6);
    double u_gap = fmax(0.0, u - 0.050);
    double r_gap = fmax(0.0, rratio_prev - 0.36);
    double pi_excess = fmax(0.0, pi - 0.050);
    double o_gap = fmax(0.0, 0.72 - O_state);

    double k_m_ease = 0.16;
    double k_t_unemp = 0.45;
    double k_o_cap = 0.22;
    double k_pi_brake = 0.30;

    double dg_m = 0.35 * (
        k_m_infra * fmax(0.0, B_infra - 0.004)
        - k_m_ease * ((B_infra < 0.003) ? (pi_excess + 0.008) : 0.0)
        - k_pi_brake * pi_excess
    );
    double dg_t = 0.0;
    if (blind_no_age != 0) {
        double u_excess = fmax(0.0, u - 0.068);
        double u_calm   = fmax(0.0, 0.064 - u);
        dg_t = 0.45 * (k_t_unemp * u_excess + k_t_age * r_gap - k_m_ease * u_calm - k_pi_brake * pi_excess);
    } else {
        double g_trans_stat = k_t_age * age_t + k_t_unemp * u_gap + 0.08 * r_gap;
        dg_t = 0.32 * (g_trans_stat - g_trans) - 0.25 * k_pi_brake * pi_excess;
    }

    double dg_o = 0.0;
    if (Reg[9] > 0.5) {
        dg_o = 0.18 * (0.045 + 0.12 * r_gap);
    } else {
        double g_org_stat = gamma_O * 0.74 + k_o_cap * o_gap + 0.05 * r_gap;
        dg_o = 0.28 * (g_org_stat - g_org);
    }

    g_maint = clampd(g_maint + clampd(dg_m, -0.015, 0.024), 0.040, clampd(Reg[4], 0.05, 0.45));
    g_trans = clampd(g_trans + clampd(dg_t, -0.015, 0.022), 0.018, clampd(Reg[5], 0.03, 0.40));
    g_org   = clampd(g_org   + clampd(dg_o, -0.008, 0.014), Reg[6], clampd(Reg[7], Reg[6], 0.22));
    double g_tot = g_maint + g_trans + g_org;

    if (Reg[3] > 0.5) {
        mu_cb = 0.0;
    } else {
        double mu_target = clampd(
            0.15 + k_cb_strain * fmax(0.0, def_rate - 0.011) + 0.20 * fmax(0.0, d_prev - 1.30) - 2.0 * pi_excess,
            0.0, Reg[8]
        );
        mu_cb = 0.65 * mu_cb + 0.35 * mu_target;
    }

    // 4. Infrastructure Fund Wear & Coercive Crowd-out
    double maint_short = fmax(0.0, g_maint_req - g_maint);
    double trans_short = fmax(0.0, g_trans_req - g_trans);
    B_infra = clampd(0.55 * B_infra + maint_short + 0.003 * fabs(fat_shock), 0.0, 0.40);
    double D_crowd = 0.65 * fmax(0.0, g_org - 0.075);

    // 5. Real Output Growth & Price Dynamics
    double g_Y_pot = 0.0;
    if (Reg[0] > 0.5) {
        double phi_cap = (is_post2024_switch && t >= t_switch)
            ? fmax(0.0, 1.0 - (double)(t - t_switch) / fmax(Reg[10], 1.0))
            : fmax(0.0, 1.0 - (double)t / fmax(Reg[10], 1.0));
        double g0 = (is_post2024_switch && t >= t_switch) ? 0.020 : 0.016;
        g_Y_pot = g0 * phi_cap * (nf_dyn / 0.92);
    } else {
        g_Y_pot = 0.023 * nf_dyn * pow(eta_safe / 0.17, 0.45);
    }

    double tau_cap = tau_max * O_state;
    double fiscal_overheat = fmax(0.0, g_tot - (tau_cap + 0.04));
    double g_Y = clampd(
        g_Y_pot - 0.22 * B_infra - 0.28 * D_crowd - 0.35 * fiscal_overheat + 0.0065 * fat_shock,
        -0.065, 0.065
    );
    double Y_real_new = fmax(1.0, Y_real * (1.0 + g_Y));

    double i_b = fmax(0.002, Reg[1] + 0.85 * pi_e - 0.018 * mu_cb);
    double i_l = i_b + 0.018 + 0.25 * def_rate;
    double i_fx = 0.020;

    pi = clampd(
        0.42 * pi_e + 0.022 * mu_cb + 0.18 * inv_eroi_dyn + 0.85 * fiscal_overheat
        + 0.18 * import_drain - 0.10 * (maint_short + trans_short) + 0.007 * eps_inf * shock_scale,
        -0.045, 0.30
    );
    pi_e = pi_e + 0.35 * (pi - pi_e);
    double P_new = fmax(0.05, P * (1.0 + pi));
    double Y_nom_new = P_new * Y_real_new;

    // 6. Exact 5-Sector Open-Economy Godley-Lavoie Accounting Update
    double NX_E = nx_tot * Y_nom_new;
    double G_maint_nom = g_maint * Y_nom_new;
    double G_org_nom   = g_org * Y_nom_new;
    double T_trans_nom = g_trans * Y_nom_new;

    double tau_eff = fmin(tau_cap, 0.94 * g_tot + 0.015 * fmax(0.0, B / fmax(Y_nom_new, 1e-6) - 0.65));
    double T_tax_nom = tau_eff * Y_nom_new;

    double Int_bh = i_b * B_h;
    double Int_lf = i_l * L_f;
    double Int_fx = i_fx * FX;

    double dB = G_maint_nom + G_org_nom + T_trans_nom + Int_bh - T_tax_nom;
    double dB_cb = (dB > 0.0) ? (mu_cb * dB) : 0.0;
    if (Reg[3] <= 0.5) {
        double qe_add = mu_cb * fmax(0.0, 0.12 * (B_h - 0.75 * Y_nom_new));
        dB_cb = fmin(dB_cb + qe_add, fmax(0.0, B_h + fmax(dB, 0.0) * 0.98));
    }
    double dB_h = dB - dB_cb;

    double dep_p = 0.045;
    double K_p_target = 2.20 * Y_real_new;
    double I_p_real_raw = fmax(0.02 * Y_real_new, dep_p * K_p + 0.25 * (K_p_target - K_p));
    double I_p_nom_raw = P_new * I_p_real_raw;
    double I_p_room = fmax(0.01 * Y_nom_new, 0.85 * Y_nom_new - G_maint_nom - G_org_nom - NX_E);
    double I_p_nom = fmin(I_p_nom_raw, I_p_room);
    double I_p_real = I_p_nom / P_new;
    double C_p_nom = Y_nom_new - I_p_nom - G_maint_nom - G_org_nom - NX_E;

    double w_org_wage = 0.65;
    double firm_gross_rev = C_p_nom + I_p_nom + G_maint_nom + (1.0 - w_org_wage) * G_org_nom + NX_E;
    double FU_f = clampd(
        0.14 * firm_gross_rev - 0.35 * Int_lf - 0.20 * (B_infra + D_crowd) * Y_nom_new,
        -0.08 * Y_nom_new, 0.25 * Y_nom_new
    );
    double Y_f_nom = firm_gross_rev - Int_lf - FU_f;
    double dL_f = I_p_nom - FU_f;

    double YD_h = Y_f_nom + w_org_wage * G_org_nom + T_trans_nom - T_tax_nom + Int_bh + Int_lf + Int_fx;
    double Sav_h = YD_h - C_p_nom;
    double dFX = NX_E + Int_fx;
    double dM_h = Sav_h - dB_h;

    double B_h_new  = B_h + dB_h;
    double B_cb_new = B_cb + dB_cb;
    double B_new    = B_h_new + B_cb_new;
    double L_f_new  = L_f + dL_f;
    double FX_new   = FX + dFX;
    double M_h_new  = M_h + dM_h;

    double K_p_new = fmax(1.0, K_p * (1.0 - dep_p) + I_p_real);
    double delta_kg = 0.04 + 0.08 * c_over_nf + clim_wear;
    double K_g_new = fmax(1.0, K_g * (1.0 - delta_kg) + g_maint * Y_real_new);

    // Strict machine-precision 5-sector closure checks
    double mon_err = fabs(M_h_new - (L_f_new + B_cb_new + FX_new)) / fmax(Y_nom_new, 1e-6);
    if (mon_err > *max_mon_err) *max_mon_err = mon_err;

    double NW_h   = M_h_new + B_h_new;
    double NW_f   = P_new * K_p_new - L_f_new;
    double NW_g   = P_new * K_g_new - B_new;
    double NW_b   = (L_f_new + B_cb_new + FX_new) - M_h_new;
    double NW_row = -FX_new;
    double nw_err = fabs((NW_h + NW_f + NW_g + NW_b + NW_row) - P_new * (K_p_new + K_g_new)) / fmax(Y_nom_new, 1e-6);
    if (nw_err > *max_nw_err) *max_nw_err = nw_err;

    if (!isfinite(mon_err) || !isfinite(nw_err) || !isfinite(phys_err) || !isfinite(Y_nom_new)) {
        *any_nan = 1;
    }

    // 7. Endogenous Soddy Solvency, External BoP Strain, Okun & Polanyian Legitimacy
    double d_priv = fmax(0.0, L_f_new) / fmax(Y_nom_new, 1e-6);
    double d_pub_mkt = fmax(0.0, B_h_new) / fmax(Y_nom_new, 1e-6);
    double d_tot = (fmax(0.0, L_f_new) + fmax(0.0, B_new)) / fmax(Y_nom_new, 1e-6);
    double bop_strain = fmax(0.0, -FX_new / fmax(Y_nom_new, 1e-6) - 0.25);

    double priv_surplus = fmax(0.10, (1.0 - g_maint - g_org) * nf_dyn);
    double r_real_mkt = fmax(0.0, i_l - pi);
    double r_real_pub = i_b - pi;
    double soddy_gap = fmax(0.0, r_real_pub - g_Y) * d_pub_mkt + fmax(0.0, (r_real_mkt - g_Y) - 0.018) * d_priv;
    double dsr = (i_l * fmax(0.0, L_f_new) + i_b * fmax(0.0, B_h_new)) / fmax(Y_nom_new, 1e-6);
    double solv_strain = clampd(
        1.35 * soddy_gap + fmax(0.0, dsr - 0.16 * priv_surplus) * 1.10 + 0.25 * bop_strain - 0.032 * mu_cb,
        0.0, 0.35
    );

    double okun_gap = fmax(0.0, Reg[2] - g_Y);
    double u_new = clampd(
        0.35 * u + 0.65 * (0.044 + 1.25 * B_infra + 0.95 * (maint_short + trans_short)
                           + 1.05 * D_crowd + 0.60 * solv_strain + 2.65 * okun_gap + 0.011 * fabs(fat_shock)),
        0.015, 0.35
    );
    double def_new = clampd(
        0.30 * def_rate + 0.70 * (0.008 + 1.40 * solv_strain + 0.45 * B_infra + 0.90 * okun_gap + 0.006 * fabs(fat_shock)),
        0.002, 0.25
    );

    double O_state_new = clampd(
        O_state + kappa_org * g_org * (1.0 - O_state)
        - (delta_ent * c_over_nf + 1.65 * solv_strain + 1.30 * B_infra) * O_state,
        0.08, 0.98
    );

    double grievance = (
        1.90 * fmax(0.0, u_new - 0.048)
        + 1.65 * trans_short
        + 1.10 * solv_strain
        + 0.80 * fmax(0.0, pi - 0.045)
    );
    double O_rev_new = clampd(
        O_rev + 1.85 * grievance * civ_t * O_rev * (1.0 - O_rev)
        - (2.15 * O_state_new * g_org + 1.35 * g_trans) * O_rev,
        0.02, 0.98
    );
    double rratio = O_rev_new / O_state_new;

    // 8. Endogenous 4-Boundary Viability Check
    double ex_l = fmax(0.0, rratio - theta_rev);
    double ex_u = fmax(0.0, u_new - u_crit);
    double ex_d = fmax(0.0, def_new - def_crit);
    double ex_p = fmax(0.0, pi - pi_crit);
    double haz = 3.20 * ex_l + 2.40 * ex_u + 2.95 * ex_d + 2.50 * ex_p;
    double p_fail = 0.26 * haz;

    if (*alive && (haz > 1e-4) && (u_haz < p_fail)) {
        *alive = 0;
        *cause = (ex_l > 0.0) ? 1 : 2;
    }
    if (*alive) *years_alive += 1.0;

    S[0] = P_new; S[1] = Y_real_new; S[2] = K_p_new; S[3] = K_g_new;
    S[4] = L_f_new; S[5] = B_h_new; S[6] = B_cb_new; S[7] = M_h_new; S[8] = FX_new;
    S[9] = C; S[10] = B_infra; S[11] = pi_e; S[12] = pi; S[13] = u_new;
    S[14] = def_new; S[15] = O_state_new; S[16] = O_rev_new;
    S[17] = g_maint; S[18] = g_trans; S[19] = g_org; S[20] = mu_cb;

    *out_nx_nom = NX_E;
    *out_g_tot  = g_tot;
    *out_d_tot  = d_tot;
    *out_rratio = rratio;
    *out_gY     = g_Y;
}


// =============================================================================
// KERNEL 1: Coupled 18-OECD + RoW Open-Economy Simulation (Mod 1)
// =============================================================================
extern "C" __global__ void kernel_mod1_open18(
    int n_threads,
    int n_countries,
    int T_steps,
    int t_switch,
    unsigned int base_seed,
    int seed_offset,
    int enable_clim_fb,
    int is_post2024_switch,
    int blind_no_age,
    double trade_absorb_frac,
    int linear_clim_mode,
    double c_mult_post,
    const double* __restrict__ prm,
    const double* __restrict__ reg_hist,
    const double* __restrict__ reg_post,
    const double* __restrict__ seq_eroi,
    const double* __restrict__ seq_eta,
    const double* __restrict__ seq_age,
    const double* __restrict__ seq_civ,
    const double* __restrict__ seq_nonf,
    const double* __restrict__ seq_net_trade,
    const double* __restrict__ seq_co2_mult,
    const double* __restrict__ ctry_init_pub_debt,
    const double* __restrict__ ctry_init_priv_debt,
    const double* __restrict__ ctry_etwh_2024,
    const double* __restrict__ ctry_co2_2024_gt,
    const double* __restrict__ ctry_clim_b15,
    const double* __restrict__ ctry_min_caps,
    double* __restrict__ out_ctry_stats,
    double* __restrict__ out_agg_traj,
    double* __restrict__ out_errs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    unsigned int world_seed = base_seed + (unsigned int)(seed_offset + tid);
    double S_all[18][25];
    double e_ref[18];
    int alive[18];
    int cause[18];
    double yrs[18];
    double last_g[18], last_d[18], last_fxr[18];

    double max_mon = 0.0, max_nw = 0.0, max_phys = 0.0, max_bop = 0.0;
    int any_nan = 0;

    for (int c = 0; c < n_countries; ++c) {
        init_country_state(S_all[c], reg_hist[9], ctry_init_pub_debt[c], ctry_init_priv_debt[c]);
        e_ref[c] = 0.0;
        alive[c] = 1;
        cause[c] = 0;
        yrs[c] = 0.0;
        last_g[c] = 0.0;
        last_d[c] = 0.0;
        last_fxr[c] = 0.0;
    }

    for (int t = 0; t < T_steps; ++t) {
        const double* reg_use = (t >= t_switch) ? reg_post : reg_hist;
        double sum_nx = 0.0;
        double sum_fx = 0.0;
        double sum_ynom = 0.0;
        double sum_g = 0.0;
        double sum_surv = 0.0;
        double sum_co2 = 0.0;
        double sum_d = 0.0;

        for (int c = 0; c < n_countries; ++c) {
            int idx = c * T_steps + t;
            int idx_p = c * T_steps + (t > 0 ? t - 1 : 0);
            double er_r = 0.0, er_i = 0.0, u_hz = 0.0;
            philox4x32_10_normals(world_seed, (unsigned int)c, (unsigned int)t, 1U, &er_r, &er_i, &u_hz);

            double nx_nom = 0.0, g_tot = 0.0, d_tot = 0.0, rratio = 0.0, g_Y = 0.0;
            step_country_sfc_bp(
                S_all[c], prm, reg_use, t, t_switch, 45,
                seq_eroi[idx], seq_eta[idx], seq_age[idx], seq_civ[idx],
                seq_nonf[idx], seq_nonf[idx_p],
                -seq_net_trade[idx],
                (t >= t_switch) ? seq_co2_mult[idx] : 1.0,
                c_mult_post,
                ctry_etwh_2024[c], ctry_co2_2024_gt[c], ctry_clim_b15[c],
                ctry_min_caps[c * 3 + 0], ctry_min_caps[c * 3 + 1], ctry_min_caps[c * 3 + 2],
                enable_clim_fb, is_post2024_switch, blind_no_age, trade_absorb_frac, linear_clim_mode,
                er_r, er_i, u_hz, &e_ref[c],
                &alive[c], &cause[c], &yrs[c],
                &nx_nom, &g_tot, &d_tot, &rratio, &g_Y,
                &max_mon, &max_nw, &max_phys, &any_nan
            );

            last_g[c] = g_tot;
            last_d[c] = d_tot;
            double y_nom_c = S_all[c][0] * S_all[c][1];
            last_fxr[c] = S_all[c][8] / fmax(y_nom_c, 1e-6);

            sum_nx += nx_nom;
            sum_fx += S_all[c][8];
            sum_ynom += y_nom_c;
            sum_g += g_tot;
            sum_surv += (double)alive[c];
            sum_co2 += S_all[c][21];
            sum_d += d_tot;
        }

        double nx_row = -sum_nx;
        double fx_row = -sum_fx;
        double bop_err = fabs(sum_nx + nx_row) / fmax(sum_ynom, 1e-6) + fabs(sum_fx + fx_row) / fmax(sum_ynom, 1e-6);
        if (bop_err > max_bop) max_bop = bop_err;

        int t_off = (tid * T_steps + t) * 5;
        out_agg_traj[t_off + 0] = sum_g / (double)n_countries;
        out_agg_traj[t_off + 1] = sum_surv / (double)n_countries;
        out_agg_traj[t_off + 2] = sum_co2;
        out_agg_traj[t_off + 3] = bop_err;
        out_agg_traj[t_off + 4] = sum_d / (double)n_countries;
    }

    for (int c = 0; c < n_countries; ++c) {
        double r_cu = S_all[c][22] / fmax(ctry_min_caps[c * 3 + 0], 1e-6);
        double r_li = S_all[c][23] / fmax(ctry_min_caps[c * 3 + 1], 1e-6);
        double r_st = S_all[c][24] / fmax(ctry_min_caps[c * 3 + 2], 1e-6);
        double mr = fmax(r_cu, fmax(r_li, r_st));
        int off = (tid * n_countries + c) * 8;
        out_ctry_stats[off + 0] = (double)alive[c];
        out_ctry_stats[off + 1] = (double)cause[c];
        out_ctry_stats[off + 2] = yrs[c];
        out_ctry_stats[off + 3] = last_g[c];
        out_ctry_stats[off + 4] = last_d[c];
        out_ctry_stats[off + 5] = S_all[c][21];
        out_ctry_stats[off + 6] = mr;
        out_ctry_stats[off + 7] = last_fxr[c];
    }

    int eoff = tid * 5;
    out_errs[eoff + 0] = max_mon;
    out_errs[eoff + 1] = max_nw;
    out_errs[eoff + 2] = max_phys;
    out_errs[eoff + 3] = max_bop;
    out_errs[eoff + 4] = (double)any_nan;
}


// =============================================================================
// KERNEL 2: ABC-SMC Particle Distance Evaluation across 18 OECD (Mod 2)
// =============================================================================
extern "C" __global__ void kernel_mod2_abc_eval(
    int n_particles,
    int n_countries,
    int T_panel,
    int t_eval,
    int n_seeds,
    unsigned int base_seed,
    const double* __restrict__ prm_mat,
    const double* __restrict__ reg_hist,
    const int*    __restrict__ ctry_mask,
    const int*    __restrict__ ctry_is_imp,
    const double* __restrict__ seq_eroi,
    const double* __restrict__ seq_eta,
    const double* __restrict__ seq_age,
    const double* __restrict__ seq_civ,
    const double* __restrict__ seq_nonf,
    const double* __restrict__ seq_net_trade,
    const double* __restrict__ obs_g,
    const double* __restrict__ obs_debt,
    const double* __restrict__ obs_pi,
    const double* __restrict__ obs_u,
    const int*    __restrict__ is_shock_yr,
    const double* __restrict__ ctry_init_pub_debt,
    const double* __restrict__ ctry_init_priv_debt,
    const double* __restrict__ ctry_etwh_2024,
    const double* __restrict__ ctry_co2_2024_gt,
    const double* __restrict__ ctry_clim_b15,
    const double* __restrict__ ctry_min_caps,
    double* __restrict__ out_dist,
    double* __restrict__ out_viab,
    double* __restrict__ out_imp_exp_gap,
    double* __restrict__ out_errs
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    double prm[22];
    for (int k = 0; k < 22; ++k) prm[k] = prm_mat[pid * 22 + k];

    double sse_g = 0.0, sse_d = 0.0, sse_pi = 0.0, sse_u = 0.0;
    double n_obs = 0.0;
    double sum_alive = 0.0;
    double n_runs = 0.0;

    // Recorded strictly as an UNOPTIMIZED emergent diagnostic (NEVER included in ABC distance!)
    double sum_g_imp = 0.0, cnt_imp = 0.0;
    double sum_g_exp = 0.0, cnt_exp = 0.0;

    double max_mon = 0.0, max_nw = 0.0, max_phys = 0.0;
    int any_nan = 0;

    for (int c = 0; c < n_countries; ++c) {
        if (ctry_mask[c] == 0) continue;
        for (int s = 0; s < n_seeds; ++s) {
            double S[25];
            init_country_state(S, reg_hist[9], ctry_init_pub_debt[c], ctry_init_priv_debt[c]);
            double e_ref = 0.0;
            int alive = 1, cause = 0;
            double yrs = 0.0;
            unsigned int seed_cs = base_seed + (unsigned int)(s * 131 + c);

            for (int t = 0; t < t_eval; ++t) {
                int idx = c * T_panel + t;
                int idx_p = c * T_panel + (t > 0 ? t - 1 : 0);
                double er_r = 0.0, er_i = 0.0, u_hz = 0.0;
                philox4x32_10_normals(seed_cs, (unsigned int)c, (unsigned int)t, 2U, &er_r, &er_i, &u_hz);

                double nx_nom = 0.0, g_tot = 0.0, d_tot = 0.0, rratio = 0.0, g_Y = 0.0;
                step_country_sfc_bp(
                    S, prm, reg_hist, t, 105, 45,
                    seq_eroi[idx], seq_eta[idx], seq_age[idx], seq_civ[idx],
                    seq_nonf[idx], seq_nonf[idx_p],
                    -seq_net_trade[idx], 1.0, 1.0,
                    ctry_etwh_2024[c], ctry_co2_2024_gt[c], ctry_clim_b15[c],
                    ctry_min_caps[c * 3 + 0], ctry_min_caps[c * 3 + 1], ctry_min_caps[c * 3 + 2],
                    0, 0, 0, 0.75, 0,
                    er_r, er_i, u_hz, &e_ref,
                    &alive, &cause, &yrs,
                    &nx_nom, &g_tot, &d_tot, &rratio, &g_Y,
                    &max_mon, &max_nw, &max_phys, &any_nan
                );

                if (is_shock_yr[idx] == 0) {
                    double dg  = (g_tot - obs_g[idx]) / 0.08;
                    double dd  = (d_tot - obs_debt[idx]) / 0.60;
                    double dpi = (S[12] - obs_pi[idx]) / 0.04;
                    double du  = (S[13] - obs_u[idx]) / 0.035;
                    sse_g  += dg * dg;
                    sse_d  += dd * dd;
                    sse_pi += dpi * dpi;
                    sse_u  += du * du;
                    n_obs  += 1.0;

                    if (t >= 50) {
                        if (ctry_is_imp[c] == 1) {
                            sum_g_imp += g_tot; cnt_imp += 1.0;
                        } else {
                            sum_g_exp += g_tot; cnt_exp += 1.0;
                        }
                    }
                }
            }
            sum_alive += (double)alive;
            n_runs += 1.0;
        }
    }

    double rmse_g  = sqrt(sse_g / fmax(n_obs, 1.0));
    double rmse_d  = sqrt(sse_d / fmax(n_obs, 1.0));
    double rmse_pi = sqrt(sse_pi / fmax(n_obs, 1.0));
    double rmse_u  = sqrt(sse_u / fmax(n_obs, 1.0));
    double diff_sim = (sum_g_imp / fmax(cnt_imp, 1.0)) - (sum_g_exp / fmax(cnt_exp, 1.0));

    // Strictly UNBIASED peacetime macro RMSE: zero target peeking on viability or importer-exporter divergence!
    out_dist[pid] = 1.50 * rmse_g + 0.55 * rmse_d + 0.45 * rmse_pi + 0.55 * rmse_u;
    out_viab[pid] = sum_alive / fmax(n_runs, 1.0);
    out_imp_exp_gap[pid] = diff_sim;

    int eoff = pid * 4;
    out_errs[eoff + 0] = max_mon;
    out_errs[eoff + 1] = max_nw;
    out_errs[eoff + 2] = max_phys;
    out_errs[eoff + 3] = (double)any_nan;
}


// =============================================================================
// KERNEL 3: Saltelli-Sobol Global Sensitivity Evaluation (Mod 3)
// =============================================================================
extern "C" __global__ void kernel_mod3_sobol_eval(
    int n_pts_batch,
    int n_crn,
    int T_steps,
    int t_switch,
    unsigned int base_seed,
    const double* __restrict__ prm_mat,
    const double* __restrict__ reg_hist,
    const double* __restrict__ seq_eroi,
    const double* __restrict__ seq_eta,
    const double* __restrict__ seq_age,
    const double* __restrict__ seq_civ,
    const double* __restrict__ seq_nonf,
    const double* __restrict__ seq_net_trade,
    const double* __restrict__ seq_co2_mult,
    double init_pub_debt,
    double init_priv_debt,
    double etwh_2024,
    double co2_2024_gt,
    double clim_b15,
    double min_cap_cu,
    double min_cap_li,
    double min_cap_st,
    double* __restrict__ out_resp,
    double* __restrict__ out_errs
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_jobs = n_pts_batch * n_crn;
    if (gid >= total_jobs) return;

    int pt = gid / n_crn;
    int crn_i = gid % n_crn;

    double prm[22];
    for (int k = 0; k < 22; ++k) prm[k] = prm_mat[pt * 22 + k];

    double S[25];
    init_country_state(S, reg_hist[9], init_pub_debt, init_priv_debt);
    double e_ref = 0.0;
    int alive = 1, cause = 0;
    double yrs = 0.0;
    double max_mon = 0.0, max_nw = 0.0, max_phys = 0.0;
    int any_nan = 0;
    double last_g = 0.0;

    unsigned int crn_seed = base_seed + (unsigned int)crn_i * 7919U;

    for (int t = 0; t < T_steps; ++t) {
        int tp = (t > 0) ? (t - 1) : 0;
        double er_r = 0.0, er_i = 0.0, u_hz = 0.0;
        philox4x32_10_normals(crn_seed, 0U, (unsigned int)t, 3U, &er_r, &er_i, &u_hz);

        double nx_nom = 0.0, g_tot = 0.0, d_tot = 0.0, rratio = 0.0, g_Y = 0.0;
        step_country_sfc_bp(
            S, prm, reg_hist, t, t_switch, 45,
            seq_eroi[t], seq_eta[t], seq_age[t], seq_civ[t],
            seq_nonf[t], seq_nonf[tp],
            -seq_net_trade[t],
            (t >= t_switch) ? seq_co2_mult[t] : 1.0,
            1.0,
            etwh_2024, co2_2024_gt, clim_b15,
            min_cap_cu, min_cap_li, min_cap_st,
            1, 1, 0, 0.75, 0,
            er_r, er_i, u_hz, &e_ref,
            &alive, &cause, &yrs,
            &nx_nom, &g_tot, &d_tot, &rratio, &g_Y,
            &max_mon, &max_nw, &max_phys, &any_nan
        );
        last_g = g_tot;
    }

    double mr = fmax(S[22] / fmax(min_cap_cu, 1e-6), fmax(S[23] / fmax(min_cap_li, 1e-6), S[24] / fmax(min_cap_st, 1e-6)));
    int roff = gid * 4;
    out_resp[roff + 0] = yrs;
    out_resp[roff + 1] = S[21];
    out_resp[roff + 2] = last_g;
    out_resp[roff + 3] = mr;

    out_errs[roff + 0] = max_mon;
    out_errs[roff + 1] = max_nw;
    out_errs[roff + 2] = max_phys;
    out_errs[roff + 3] = (double)any_nan;
}


// =============================================================================
// KERNEL 4: 4D Construction-Phase Cannibalization x Minerals x Post-Growth (Mod 4)
// =============================================================================
extern "C" __global__ void kernel_mod4_cannib_minerals(
    int n_grid_pts,
    int n_seeds,
    int T_steps,
    int t_switch,
    unsigned int base_seed,
    const double* __restrict__ base_prm,
    const double* __restrict__ grid_cfg,
    const double* __restrict__ reg_growth,
    const double* __restrict__ reg_pg,
    const double* __restrict__ seq_eroi_hist,
    const double* __restrict__ seq_eta_hist,
    const double* __restrict__ seq_age,
    const double* __restrict__ seq_civ,
    const double* __restrict__ seq_nonf_hist,
    const double* __restrict__ seq_net_trade,
    double etwh_2024,
    double co2_2024_gt,
    double clim_b15,
    double min_cap_cu,
    double min_cap_li,
    double min_cap_st,
    double* __restrict__ out_res,
    double* __restrict__ out_nan
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_grid_pts * n_seeds) return;

    int gidx = gid / n_seeds;
    int sidx = gid % n_seeds;

    double prm[22];
    for (int k = 0; k < 22; ++k) prm[k] = base_prm[k];

    double nonf_tgt     = grid_cfg[gidx * 4 + 0];
    double eta_rec_val  = grid_cfg[gidx * 4 + 1];
    double lam_learn    = grid_cfg[gidx * 4 + 2];
    int    is_pg        = (grid_cfg[gidx * 4 + 3] > 0.5) ? 1 : 0;

    prm[11] = lam_learn;
    prm[12] = eta_rec_val;

    double S[25];
    init_country_state(S, reg_growth[9], 0.47, 0.48);
    double e_ref = 0.0;
    int alive = 1, cause = 0;
    double yrs = 0.0;
    double max_mon = 0.0, max_nw = 0.0, max_phys = 0.0;
    int any_nan = 0;

    double nf_prev = seq_nonf_hist[0];
    double nf_2024 = seq_nonf_hist[t_switch - 1];
    double eroi_24 = seq_eroi_hist[t_switch - 1];
    double eta_24  = seq_eta_hist[t_switch - 1];
    double eroi_tgt = 14.0 + 10.0 * nonf_tgt - 2.0 * (lam_learn - 0.70);
    double eta_tgt  = eta_24 + 0.14 * (nonf_tgt - nf_2024);

    double peak_build = 0.0;
    unsigned int sc = base_seed + (unsigned int)sidx;

    for (int t = 0; t < T_steps; ++t) {
        double eroi_t, eta_t, nonf_t, co2_mult_t, c_mult_t;
        if (t < t_switch) {
            eroi_t = seq_eroi_hist[t];
            eta_t  = seq_eta_hist[t];
            nonf_t = seq_nonf_hist[t];
            co2_mult_t = 1.0;
            c_mult_t = 1.0;
        } else {
            double frac = (double)(t - t_switch + 1) / (double)(T_steps - t_switch);
            double ramp = is_pg ? (1.0 - exp(-2.8 * frac)) / (1.0 - exp(-2.8)) : pow(frac, 0.90);
            nonf_t = nf_2024 + (nonf_tgt - nf_2024) * ramp;
            eroi_t = eroi_24 + (eroi_tgt - eroi_24) * frac;
            eta_t  = eta_24  + (eta_tgt - eta_24) * frac;
            co2_mult_t = (1.0 - nonf_t) / fmax(1.0 - nf_2024, 0.10);
            c_mult_t = is_pg ? 0.50 : 0.90;
        }

        double d_nf = clampd(nonf_t - nf_prev, 0.0, 0.045);
        double vre_rep = (t >= t_switch) ? (fmax(0.0, nonf_t - 0.16) / 35.0) : 0.0;
        double b_rate = (t >= 45) ? fmin(0.065, d_nf + vre_rep) : 0.0;
        double b_est = (t >= 45) ? fmin(0.22, prm[10] * lam_learn * b_rate) : 0.0;
        if (t >= t_switch && b_est > peak_build) peak_build = b_est;

        double er_r = 0.0, er_i = 0.0, u_hz = 0.0;
        philox4x32_10_normals(sc, (unsigned int)gidx, (unsigned int)t, 4U, &er_r, &er_i, &u_hz);

        const double* reg_u = (is_pg && t >= t_switch) ? reg_pg : reg_growth;
        double nx_nom = 0.0, g_tot = 0.0, d_tot = 0.0, rratio = 0.0, g_Y = 0.0;
        step_country_sfc_bp(
            S, prm, reg_u, t, t_switch, 45,
            eroi_t, eta_t, seq_age[t], seq_civ[t],
            nonf_t, nf_prev,
            -seq_net_trade[t], co2_mult_t, c_mult_t,
            etwh_2024, co2_2024_gt, clim_b15,
            min_cap_cu, min_cap_li, min_cap_st,
            1, 1, 0, 0.75, 0,
            er_r, er_i, u_hz, &e_ref,
            &alive, &cause, &yrs,
            &nx_nom, &g_tot, &d_tot, &rratio, &g_Y,
            &max_mon, &max_nw, &max_phys, &any_nan
        );
        nf_prev = nonf_t;
    }

    double mr = fmax(S[22] / fmax(min_cap_cu, 1e-6), fmax(S[23] / fmax(min_cap_li, 1e-6), S[24] / fmax(min_cap_st, 1e-6)));
    double merr = fmax(max_mon, fmax(max_nw, max_phys));
    int o = gid * 6;
    out_res[o + 0] = (double)alive;
    out_res[o + 1] = S[21];
    out_res[o + 2] = mr;
    out_res[o + 3] = peak_build;
    out_res[o + 4] = yrs;
    out_res[o + 5] = merr;
    out_nan[gid] = (double)any_nan;
}


// =============================================================================
// KERNEL 5A: Ultra-Dense Phase Diagram + Early Warning Signals (Mod 5A)
// =============================================================================
extern "C" __global__ void kernel_mod5_phase_ews(
    int n_pts_batch,
    int n_seeds,
    int n_steps,
    unsigned int base_seed,
    double impdep_signed_fix,
    const double* __restrict__ prm,
    const double* __restrict__ eroi_vec,
    const double* __restrict__ gorg_vec,
    double* __restrict__ out_metrics,
    double* __restrict__ out_nan
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_pts_batch * n_seeds) return;

    int pid = gid / n_seeds;
    int sid = gid % n_seeds;
    double eroi_fix = eroi_vec[pid];
    double gorg_fix = gorg_vec[pid];

    double civ_budget = clampd(0.435 - gorg_fix, 0.12, 0.39);
    double reg[11] = {
        0.0, 0.022, 0.012, 0.0,
        civ_budget * 0.52, civ_budget * 0.48,
        gorg_fix, gorg_fix, 0.85,
        (gorg_fix >= 0.085) ? 1.0 : 0.0, 30.0
    };

    double S[25];
    init_country_state(S, reg[9], 0.85, 1.10);
    S[4] = 110.0; S[5] = 65.0; S[6] = 20.0; S[7] = S[4] + S[6]; S[8] = 0.0;
    S[9] = 0.72;  S[15] = 0.64; S[16] = 0.16; S[20] = 0.30;
    S[17] = civ_budget * 0.52;
    S[18] = civ_budget * 0.48;
    S[19] = gorg_fix;

    double e_ref = 0.0;
    int alive = 1, cause = 0;
    double yrs = 0.0;
    double max_mon = 0.0, max_nw = 0.0, max_phys = 0.0;
    int any_nan = 0;

    double d_win[45], b_win[45];
    int w_cnt = 0;

    unsigned int sc = base_seed + (unsigned int)sid;
    for (int t = 0; t < n_steps; ++t) {
        double er_r = 0.0, er_i = 0.0, u_hz = 0.0;
        philox4x32_10_normals(sc, (unsigned int)pid, (unsigned int)t, 5U, &er_r, &er_i, &u_hz);

        int was_alive = alive;
        double nx_nom = 0.0, g_tot = 0.0, d_tot = 0.0, rratio = 0.0, g_Y = 0.0;
        step_country_sfc_bp(
            S, prm, reg, t, n_steps + 5, 0,
            eroi_fix, 0.255, 0.30, 1.05, 0.14, 0.14,
            impdep_signed_fix, 1.0, 1.0,
            1500.0, 0.5, 50.0, 1e6, 1e6, 1e6,
            0, 0, 0, 0.75, 0,
            er_r, er_i, u_hz, &e_ref,
            &alive, &cause, &yrs,
            &nx_nom, &g_tot, &d_tot, &rratio, &g_Y,
            &max_mon, &max_nw, &max_phys, &any_nan
        );
        // Record pre-collapse EWS window (up to the final alive horizon)
        if (was_alive && t >= 5 && w_cnt < 45) {
            d_win[w_cnt] = d_tot;
            b_win[w_cnt] = S[10];
            w_cnt++;
        }
    }

    // Use first-differenced debt (Δd_tot) & B_infra level over the final <=20 alive steps for genuine CSD EWS
    int i0_w = (w_cnt > 20) ? (w_cnt - 20) : 0;
    int m_len = w_cnt - i0_w;
    double mb = 0.0, md = 0.0;
    int dn_cnt = (m_len > 1) ? (m_len - 1) : 1;
    for (int i = i0_w; i < w_cnt; ++i) { mb += b_win[i]; }
    mb /= fmax((double)m_len, 1.0);
    for (int i = i0_w + 1; i < w_cnt; ++i) { md += (d_win[i] - d_win[i - 1]); }
    md /= fmax((double)dn_cnt, 1.0);

    double vb = 0.0, c1b = 0.0, vd = 0.0, c1d = 0.0;
    for (int i = i0_w; i < w_cnt; ++i) {
        double bb = b_win[i] - mb;
        vb += bb * bb;
        if (i > i0_w) {
            c1b += bb * (b_win[i - 1] - mb);
            double dd = (d_win[i] - d_win[i - 1]) - md;
            vd += dd * dd;
            if (i > i0_w + 1) {
                double dd_prev = (d_win[i - 1] - d_win[i - 2]) - md;
                c1d += dd * dd_prev;
            }
        }
    }
    double var_b = vb / fmax((double)m_len, 1.0);
    double ar1_d = (vd > 1e-12) ? clampd(c1d / vd, -0.99, 0.99) : 0.0;
    double ar1_b = (vb > 1e-12) ? clampd(c1b / vb, -0.99, 0.99) : 0.0;
    double merr = fmax(max_mon, fmax(max_nw, max_phys));

    int o = gid * 6;
    out_metrics[o + 0] = (double)alive;
    out_metrics[o + 1] = (double)cause;
    out_metrics[o + 2] = ar1_d;
    out_metrics[o + 3] = var_b;
    out_metrics[o + 4] = ar1_b;
    out_metrics[o + 5] = merr;
    out_nan[gid] = (double)any_nan;
}


// =============================================================================
// KERNEL 5B: Irreversible Hysteresis Loop (Ramp EROI 25 -> 6 -> 25 over 80 yrs)
// =============================================================================
extern "C" __global__ void kernel_mod5_hysteresis(
    int n_seeds,
    int n_steps,
    unsigned int base_seed,
    double impdep_signed_fix,
    const double* __restrict__ prm,
    const double* __restrict__ reg,
    const double* __restrict__ seq_eroi,
    double* __restrict__ out_step_stats,
    double* __restrict__ out_errs
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_seeds) return;

    double S[25];
    init_country_state(S, reg[9], 0.55, 0.48);
    double e_ref = 0.0;
    int alive = 1, cause = 0;
    double yrs = 0.0;
    double max_mon = 0.0, max_nw = 0.0, max_phys = 0.0;
    int any_nan = 0;
    unsigned int sc = base_seed + (unsigned int)sid;

    for (int t = 0; t < n_steps; ++t) {
        double er_r = 0.0, er_i = 0.0, u_hz = 0.0;
        philox4x32_10_normals(sc, 99U, (unsigned int)t, 6U, &er_r, &er_i, &u_hz);
        double nx_nom = 0.0, g_tot = 0.0, d_tot = 0.0, rratio = 0.0, g_Y = 0.0;
        step_country_sfc_bp(
            S, prm, reg, t, n_steps + 5, 0,
            seq_eroi[t], 0.255, 0.28, 1.00, 0.14, 0.14,
            impdep_signed_fix, 1.0, 1.0,
            1500.0, 0.5, 50.0, 1e6, 1e6, 1e6,
            0, 0, 0, 0.75, 0,
            er_r, er_i, u_hz, &e_ref,
            &alive, &cause, &yrs,
            &nx_nom, &g_tot, &d_tot, &rratio, &g_Y,
            &max_mon, &max_nw, &max_phys, &any_nan
        );
        int o = (sid * n_steps + t) * 4;
        out_step_stats[o + 0] = (double)alive;
        out_step_stats[o + 1] = S[15];
        out_step_stats[o + 2] = S[10];
        out_step_stats[o + 3] = g_tot;
    }
    int eo = sid * 4;
    out_errs[eo + 0] = max_mon;
    out_errs[eo + 1] = max_nw;
    out_errs[eo + 2] = max_phys;
    out_errs[eo + 3] = (double)any_nan;
}


// =============================================================================
// KERNEL 6: China Out-of-Sample Zero-Leakage Posterior Ensemble (1950-2050, Mod 6)
// =============================================================================
extern "C" __global__ void kernel_mod6_china_oos(
    int n_particles_batch,
    int n_seeds_per_p,
    int T_steps,     // 101 (1950-2050)
    int t_switch,    // 75  (year 2025)
    unsigned int base_seed,
    double override_lam_learn,
    double override_eta_rec,
    double c_mult_post,
    int enable_clim_fb,
    int is_post2024_switch,
    int blind_no_age,
    double trade_absorb_frac,
    int linear_clim_mode,
    const double* __restrict__ prm_mat,
    const double* __restrict__ reg_hist,
    const double* __restrict__ reg_post,
    const double* __restrict__ seq_eroi,
    const double* __restrict__ seq_eta,
    const double* __restrict__ seq_age,
    const double* __restrict__ seq_civ,
    const double* __restrict__ seq_nonf,
    const double* __restrict__ seq_net_trade,
    const double* __restrict__ seq_co2_mult,
    double init_pub_debt,
    double init_priv_debt,
    double etwh_2024,
    double co2_2024_gt,
    double clim_budget_chn,
    double min_cap_cu,
    double min_cap_li,
    double min_cap_st,
    int write_full_traj,
    double* __restrict__ out_summary,
    double* __restrict__ out_traj,
    double* __restrict__ out_errs
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_jobs = n_particles_batch * n_seeds_per_p;
    if (gid >= total_jobs) return;

    int pid = gid / n_seeds_per_p;
    int sid = gid % n_seeds_per_p;

    double prm[22];
    for (int k = 0; k < 22; ++k) prm[k] = prm_mat[pid * 22 + k];
    if (override_lam_learn > 0.0) prm[11] = override_lam_learn;
    if (override_eta_rec >= 0.0)  prm[12] = override_eta_rec;

    double S[25];
    init_country_state(S, reg_hist[9], init_pub_debt, init_priv_debt);
    S[17] = 0.085; S[18] = 0.025; S[19] = 0.055;

    double e_ref = 0.0;
    int alive = 1, cause = 0;
    double yrs = 0.0;
    double max_mon = 0.0, max_nw = 0.0, max_phys = 0.0;
    int any_nan = 0;
    double last_g = 0.0, last_d = 0.0;

    unsigned int sc = base_seed + (unsigned int)sid * 1009U;
    for (int t = 0; t < T_steps; ++t) {
        int tp = (t > 0) ? (t - 1) : 0;
        const double* reg_u = (t >= t_switch) ? reg_post : reg_hist;
        double er_r = 0.0, er_i = 0.0, u_hz = 0.0;
        philox4x32_10_normals(sc, 19U, (unsigned int)t, 7U, &er_r, &er_i, &u_hz);

        double nx_nom = 0.0, g_tot = 0.0, d_tot = 0.0, rratio = 0.0, g_Y = 0.0;
        step_country_sfc_bp(
            S, prm, reg_u, t, t_switch, 15,
            seq_eroi[t], seq_eta[t], seq_age[t], seq_civ[t],
            seq_nonf[t], seq_nonf[tp],
            -seq_net_trade[t],
            (t >= t_switch) ? seq_co2_mult[t] : 1.0,
            c_mult_post,
            etwh_2024, co2_2024_gt, clim_budget_chn,
            min_cap_cu, min_cap_li, min_cap_st,
            enable_clim_fb, is_post2024_switch, blind_no_age, trade_absorb_frac, linear_clim_mode,
            er_r, er_i, u_hz, &e_ref,
            &alive, &cause, &yrs,
            &nx_nom, &g_tot, &d_tot, &rratio, &g_Y,
            &max_mon, &max_nw, &max_phys, &any_nan
        );
        last_g = g_tot;
        last_d = d_tot;
        if (write_full_traj) {
            int toff = (gid * T_steps + t) * 4;
            out_traj[toff + 0] = g_tot;
            out_traj[toff + 1] = d_tot;
            out_traj[toff + 2] = (double)alive;
            out_traj[toff + 3] = S[21];
        }
    }

    double mr = fmax(S[22] / fmax(min_cap_cu, 1e-6), fmax(S[23] / fmax(min_cap_li, 1e-6), S[24] / fmax(min_cap_st, 1e-6)));
    double y_nom_end = S[0] * S[1];
    int soff = gid * 8;
    out_summary[soff + 0] = (double)alive;
    out_summary[soff + 1] = (double)cause;
    out_summary[soff + 2] = yrs;
    out_summary[soff + 3] = last_g;
    out_summary[soff + 4] = last_d;
    out_summary[soff + 5] = S[21];
    out_summary[soff + 6] = mr;
    out_summary[soff + 7] = S[8] / fmax(y_nom_end, 1e-6);

    int eoff = gid * 4;
    out_errs[eoff + 0] = max_mon;
    out_errs[eoff + 1] = max_nw;
    out_errs[eoff + 2] = max_phys;
    out_errs[eoff + 3] = (double)any_nan;
}

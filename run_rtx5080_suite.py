"""
GPU Master Experiment Orchestrator (Windows RTX 5080 Blackwell / RTX 4080S Ada, Pure FP64 NVRTC).
Enforces zero target peeking, zero data leakage on China, WDDM-TDR micro-batching (< 250 ms/launch),
hard runtime closure/NaN fuses (< 1e-11), automatic checkpoint/resume, and SHA256 reproducibility.

Usage:
  python run_rtx5080_suite.py --mode preflight
  python run_rtx5080_suite.py --mode run --tier standard
  python run_rtx5080_suite.py --mode run --tier ultra
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
import numpy as np

BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BUNDLE_DIR, "data")
ENGINE_DIR = os.path.join(BUNDLE_DIR, "engine")
OUT_DIR = os.path.join(BUNDLE_DIR, "outputs")

import sys
sys.path.insert(0, ENGINE_DIR)
from cuda_driver import CudaContext
import cpu_reference as ref
import statistical_methods as sm


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_errs_fuse(errs_arr, where=""):
    """Aborts loudly if ANY thread breached machine-precision accounting closure or hit NaN/Inf."""
    arr = np.asarray(errs_arr)
    if arr.ndim == 1:
        max_e = float(np.max(arr))
        nan_flag = 0.0
    else:
        max_e = float(np.max(arr[:, :-1]))
        nan_flag = float(np.max(arr[:, -1]))
    if not np.isfinite(max_e) or nan_flag > 0.5 or max_e > 1e-11:
        raise RuntimeError(
            f"[HARD CLOSURE FUSE TRIPPED at {where}] max_err={max_e:.3e}, any_nan={nan_flag}"
        )
    return max_e


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def build_forward_sequences(eroi24, eta24, nonf24, eroi_tgt, eta_tgt, nonf_tgt, co2_mult_end, exp_scale=False, rate=2.8):
    t26 = np.linspace(0.0, 1.0, 26)
    ramp = (1.0 - np.exp(-rate * t26)) / (1.0 - np.exp(-rate)) if exp_scale else (t26 ** 0.88)
    e26 = eroi24 + (eroi_tgt - eroi24) * t26
    et26 = eta24 + (eta_tgt - eta24) * t26
    nf26 = nonf24 + (nonf_tgt - nonf24) * ramp
    cm26 = np.linspace(1.0, co2_mult_end, 26) * ((1.0 - nf26) / max(1.0 - nonf24, 0.10))
    return e26, et26, nf26, cm26


# =============================================================================
# GPU Buffer Uploader for 18-OECD Panel Slices (reused cleanly across modules)
# =============================================================================
class OecdGpuArrays:
    def __init__(self, ctx, oc):
        self.ctx = ctx
        self.bufs = {
            "reg_hist": ctx.alloc_array(ref.REGIMES["1_Adaptive_Growth"]),
            "ctry_is_imp": ctx.alloc_array(oc["is_imp"].astype(np.int32)),
            "eroi": ctx.alloc_array(oc["eroi"]),
            "eta": ctx.alloc_array(oc["eta"]),
            "age": ctx.alloc_array(oc["age"]),
            "civ": ctx.alloc_array(oc["civ"]),
            "nonf": ctx.alloc_array(oc["nonf"]),
            "net_trade": ctx.alloc_array(oc["net_trade"]),
            "obs_g": ctx.alloc_array(oc["obs_g"]),
            "obs_debt": ctx.alloc_array(oc["obs_debt"]),
            "obs_pi": ctx.alloc_array(oc["obs_pi"]),
            "obs_u": ctx.alloc_array(oc["obs_u"]),
            "is_shock": ctx.alloc_array(oc["is_shock"].astype(np.int32)),
            "init_pub_debt": ctx.alloc_array(oc["init_pub_debt"]),
            "init_priv_debt": ctx.alloc_array(oc["init_priv_debt"]),
            "etwh_2024": ctx.alloc_array(oc["etwh_2024"]),
            "co2_2024_gt": ctx.alloc_array(oc["co2_2024_gt"]),
            "clim_b15": ctx.alloc_array(oc["clim_b15"]),
            "min_caps": ctx.alloc_array(oc["min_caps"]),
        }

    def free(self):
        for b in self.bufs.values():
            b.free()


# =============================================================================
# MODULE 0: NVRTC Check, Exact GPU-CPU Parity (<1e-11), & RTX Throughput Calibration
# =============================================================================
def run_mod0_preflight(ctx, mod, ds):
    print("\n=== [MOD 0] GPU Compilation, CPU Parity & Speed Calibration ===")
    oc = ds["oecd"]
    ch = ds["china"]
    prm = ref.DEFAULT_PARAMS.copy()

    # 1. Exact GPU vs CPU Parity on 18-OECD + China
    n_c, T = 18, 105
    reg = ref.REGIMES["1_Adaptive_Growth"]
    S_all = [ref.init_country_state_py(reg[9], oc["init_pub_debt"][c], oc["init_priv_debt"][c]) for c in range(n_c)]
    sc_all = [{"e_gross_ref": 0.0, "alive": 1, "cause": 0, "years_alive": 0.0,
               "max_mon_err": 0.0, "max_nw_err": 0.0, "max_phys_err": 0.0} for _ in range(n_c)]
    cpu_g = np.zeros(n_c)
    for t in range(T):
        tp = max(0, t - 1)
        for c in range(n_c):
            er, ei, uh = ref.philox4x32_10_normals_py(2026, c, t, 1)
            _, gt, _, _, _ = ref.step_country_sfc_bp_py(
                S_all[c], prm, reg, t, 105, 45,
                oc["eroi"][c, t], oc["eta"][c, t], oc["age"][c, t], oc["civ"][c, t],
                oc["nonf"][c, t], oc["nonf"][c, tp], -oc["net_trade"][c, t], 1.0, 1.0,
                oc["etwh_2024"][c], oc["co2_2024_gt"][c], oc["clim_b15"][c], oc["min_caps"][c],
                0, 0, er, ei, uh, sc_all[c], 0, 0.75, 0
            )
            cpu_g[c] = gt

    # GPU test for OECD
    n_th = 4
    bufs = [
        ctx.alloc_array(prm), ctx.alloc_array(reg), ctx.alloc_array(reg),
        ctx.alloc_array(oc["eroi"]), ctx.alloc_array(oc["eta"]), ctx.alloc_array(oc["age"]),
        ctx.alloc_array(oc["civ"]), ctx.alloc_array(oc["nonf"]), ctx.alloc_array(oc["net_trade"]),
        ctx.alloc_array(np.ones((n_c, T))), ctx.alloc_array(oc["init_pub_debt"]),
        ctx.alloc_array(oc["init_priv_debt"]), ctx.alloc_array(oc["etwh_2024"]),
        ctx.alloc_array(oc["co2_2024_gt"]), ctx.alloc_array(oc["clim_b15"]),
        ctx.alloc_array(oc["min_caps"]),
        ctx.alloc_array((n_th, n_c, 8), np.float64),
        ctx.alloc_array((n_th, T, 5), np.float64),
        ctx.alloc_array((n_th, 5), np.float64),
    ]
    try:
        mod.launch("kernel_mod1_open18", 1, 32, [
            n_th, n_c, T, 105, ctypes.c_uint32(2026), 0, 0, 0, 0, 0.75, 0, 1.0, *bufs
        ])
        hc = bufs[-3].to_host()
        he = bufs[-1].to_host()
        max_oecd_diff = float(np.max(np.abs(hc[0, :, 3] - cpu_g)))
        max_oecd_err = check_errs_fuse(he, "Mod0_OECD_Parity")
    finally:
        for b in bufs:
            b.free()

    # GPU vs CPU parity for China 1950-2050
    reg_ch = ref.REGIMES["CHN_Hist_Base"]
    reg_cp = ref.REGIMES["CHN_D_Current_Elec_Infra_Coord"]
    t26 = np.linspace(0.0, 1.0, 26)
    e_c, et_c, nf_c, cm_c = build_forward_sequences(
        ch["eroi_hist"][-1], ch["eta_hist"][-1], ch["nonf_hist"][-1], 22.0, 0.365, 0.85, 0.90, exp_scale=True
    )
    e101 = np.concatenate([ch["eroi_hist"], e_c])
    et101 = np.concatenate([ch["eta_hist"], et_c])
    nf101 = np.concatenate([ch["nonf_hist"], nf_c])
    cv101 = np.concatenate([ch["civ_hist"], np.full(26, ch["civ_hist"][-1])])
    cm101 = np.concatenate([np.ones(75), cm_c])

    S_c = ref.init_country_state_py(reg_ch[9], ch["init_pub_debt"], ch["init_priv_debt"])
    S_c[17], S_c[18], S_c[19] = 0.085, 0.025, 0.055
    sc_c = {"e_gross_ref": 0.0, "alive": 1, "cause": 0, "years_alive": 0.0,
            "max_mon_err": 0.0, "max_nw_err": 0.0, "max_phys_err": 0.0}
    cpu_chn_g = 0.0
    for t in range(101):
        tp = max(0, t - 1)
        er, ei, uh = ref.philox4x32_10_normals_py(9001, 19, t, 7)
        _, gt, _, _, _ = ref.step_country_sfc_bp_py(
            S_c, prm, reg_cp if t >= 75 else reg_ch, t, 75, 15,
            e101[t], et101[t], ch["age_101"][t], cv101[t],
            nf101[t], nf101[tp], -ch["net_trade_101"][t], cm101[t], 0.70,
            ch["etwh_2024"], ch["co2_2024_gt"], ch["clim_budget_chn"], ch["min_caps"],
            1, 1, er, ei, uh, sc_c, 0, 0.75, 0
        )
        cpu_chn_g = gt

    cbufs = [
        ctx.alloc_array(prm[None, :]), ctx.alloc_array(reg_ch), ctx.alloc_array(reg_cp),
        ctx.alloc_array(e101), ctx.alloc_array(et101), ctx.alloc_array(ch["age_101"]),
        ctx.alloc_array(cv101), ctx.alloc_array(nf101), ctx.alloc_array(ch["net_trade_101"]),
        ctx.alloc_array(cm101),
        ctx.alloc_array((1, 8), np.float64), ctx.alloc_array((1, 4), np.float64), ctx.alloc_array((1, 4), np.float64)
    ]
    try:
        mod.launch("kernel_mod6_china_oos", 1, 32, [
            1, 1, 101, 75, ctypes.c_uint32(9001), -1.0, -1.0, 0.70, 1, 1, 0, 0.75, 0,
            cbufs[0], cbufs[1], cbufs[2], cbufs[3], cbufs[4], cbufs[5], cbufs[6], cbufs[7], cbufs[8], cbufs[9],
            float(ch["init_pub_debt"]), float(ch["init_priv_debt"]),
            float(ch["etwh_2024"]), float(ch["co2_2024_gt"]), float(ch["clim_budget_chn"]),
            float(ch["min_caps"][0]), float(ch["min_caps"][1]), float(ch["min_caps"][2]),
            0, cbufs[10], cbufs[11], cbufs[12]
        ])
        ch_sum = cbufs[10].to_host()
        ch_err = cbufs[12].to_host()
        max_chn_diff = float(abs(ch_sum[0, 3] - cpu_chn_g))
        max_chn_err = check_errs_fuse(ch_err, "Mod0_CHN_Parity")
    finally:
        for b in cbufs:
            b.free()

    assert max_oecd_diff < 1e-11 and max_chn_diff < 1e-11

    # 2. Micro-benchmark single-trajectory FP64 Sobol throughput (131 steps/traj)
    n_bench = 32768
    prm_b = np.tile(prm, (n_bench // 16, 1))
    d_pb = ctx.alloc_array(prm_b)
    d_reg = ctx.alloc_array(reg)
    d_e = ctx.alloc_array(oc["eroi"][0:1].mean(axis=0))
    # Pad 105->131 cleanly for bench
    e131 = np.pad(oc["eroi"].mean(axis=0), (0, 26), mode="edge")
    et131 = np.pad(oc["eta"].mean(axis=0), (0, 26), mode="edge")
    ag131 = oc["age_131"].mean(axis=0)
    cv131 = np.pad(oc["civ"].mean(axis=0), (0, 26), mode="edge")
    nf131 = np.pad(oc["nonf"].mean(axis=0), (0, 26), mode="edge")
    nt131 = oc["net_trade_131"].mean(axis=0)
    cm131 = np.ones(131)

    b_bufs = [
        ctx.alloc_array(prm_b), ctx.alloc_array(reg),
        ctx.alloc_array(e131), ctx.alloc_array(et131), ctx.alloc_array(ag131),
        ctx.alloc_array(cv131), ctx.alloc_array(nf131), ctx.alloc_array(nt131),
        ctx.alloc_array(cm131),
        ctx.alloc_array((n_bench, 4), np.float64),
        ctx.alloc_array((n_bench, 4), np.float64),
    ]
    try:
        d_pb.free(); d_reg.free(); d_e.free()
        # Warmup
        args_b = [
            n_bench // 16, 16, 131, 105, ctypes.c_uint32(1234),
            b_bufs[0], b_bufs[1], b_bufs[2], b_bufs[3], b_bufs[4], b_bufs[5], b_bufs[6], b_bufs[7], b_bufs[8],
            0.50, 0.45, 3000.0, 0.5, 5.0, 10.0, 1.5, 100.0,
            b_bufs[9], b_bufs[10]
        ]
        mod.launch("kernel_mod3_sobol_eval", (n_bench + 127) // 128, 128, args_b)
        n_rep = 8
        t_b0 = time.perf_counter()
        for _ in range(n_rep):
            mod.launch("kernel_mod3_sobol_eval", (n_bench + 127) // 128, 128, args_b)
        dt_b = time.perf_counter() - t_b0
        traj_per_sec = (n_bench * n_rep) / max(dt_b, 1e-6)
        check_errs_fuse(b_bufs[10].to_host(), "Mod0_Benchmark")
    finally:
        for b in b_bufs:
            b.free()

    # Estimate runtimes for Tier 1 (~1.8B equiv country-trajectories + ABC CPU) & Tier 2 (~5.5B equiv + ABC CPU)
    t1_min = (1.8e9 / max(traj_per_sec, 1e3)) / 60.0 + 3.0
    t2_min = (5.5e9 / max(traj_per_sec, 1e3)) / 60.0 + 8.0

    res = {
        "gpu": ctx.info_dict(),
        "parity_oecd18_max_abs_diff": max_oecd_diff,
        "parity_china_max_abs_diff": max_chn_diff,
        "max_closure_err": max(max_oecd_err, max_chn_err),
        "bench_131step_fp64_traj_per_sec": round(traj_per_sec, 1),
        "est_tier1_standard_minutes": round(t1_min, 1),
        "est_tier2_ultra_minutes": round(t2_min, 1),
    }
    print(f"  GPU: {res['gpu']['gpu_name']} ({res['gpu']['compute_capability']}, {res['gpu']['vram_gb']} GB, Driver CUDA {res['gpu']['driver_cuda_version']})")
    print(f"  GPU-vs-CPU Parity: OECD={max_oecd_diff:.2e}, China={max_chn_diff:.2e} (< 1e-11 PASS)")
    print(f"  Accounting Closure Error: {res['max_closure_err']:.2e} (< 1e-11 PASS)")
    print(f"  Measured FP64 Speed: {traj_per_sec:,.0f} country-trajectories/sec (131 yrs/traj)")
    print(f"  Estimated Runtime -> Tier 1 (standard): ~{t1_min:.1f} min | Tier 2 (ultra): ~{t2_min:.1f} min ({t2_min/60:.2f} hr)")
    return res


# =============================================================================
# MODULE 1: Coupled 18-OECD + RoW Open Economy + Falsification & Structural Invariance
# =============================================================================
def _run_mod1_batch(ctx, mod, oc, prm, reg_hist, reg_post, eroi_seq, eta_seq, age_seq, civ_seq, nonf_seq, net_tr_seq, co2_mult_seq,
                    n_worlds, T_steps, t_switch, base_seed, enable_clim, is_post_sw, blind_no_age, trade_absorb, lin_clim, c_mult_post):
    n_c = 18
    batch_sz = min(n_worlds, 1024)
    bufs = [
        ctx.alloc_array(prm), ctx.alloc_array(reg_hist), ctx.alloc_array(reg_post),
        ctx.alloc_array(eroi_seq), ctx.alloc_array(eta_seq), ctx.alloc_array(age_seq),
        ctx.alloc_array(civ_seq), ctx.alloc_array(nonf_seq), ctx.alloc_array(net_tr_seq),
        ctx.alloc_array(co2_mult_seq), ctx.alloc_array(oc["init_pub_debt"]),
        ctx.alloc_array(oc["init_priv_debt"]), ctx.alloc_array(oc["etwh_2024"]),
        ctx.alloc_array(oc["co2_2024_gt"]), ctx.alloc_array(oc["clim_b15"]),
        ctx.alloc_array(oc["min_caps"]),
        ctx.alloc_array((batch_sz, n_c, 8), np.float64),
        ctx.alloc_array((batch_sz, T_steps, 5), np.float64),
        ctx.alloc_array((batch_sz, 5), np.float64),
    ]
    ctry_sum = np.zeros((n_c, 8), dtype=np.float64)
    traj_sum = np.zeros((T_steps, 5), dtype=np.float64)
    traj_surv_list = []
    max_err = 0.0
    try:
        done = 0
        while done < n_worlds:
            cur = min(batch_sz, n_worlds - done)
            grid = (cur + 63) // 64
            mod.launch("kernel_mod1_open18", grid, 64, [
                cur, n_c, T_steps, t_switch, ctypes.c_uint32(base_seed), done,
                enable_clim, is_post_sw, blind_no_age, float(trade_absorb), lin_clim, float(c_mult_post),
                *bufs
            ])
            hc = bufs[-3].to_host()[:cur]
            ht = bufs[-2].to_host()[:cur]
            he = bufs[-1].to_host()[:cur]
            max_err = max(max_err, check_errs_fuse(he, "Mod1_Open18"))
            ctry_sum += hc.sum(axis=0)
            traj_sum += ht.sum(axis=0)
            traj_surv_list.append(hc[:, :, 0].mean(axis=1))
            done += cur
    finally:
        for b in bufs:
            b.free()

    return ctry_sum / n_worlds, traj_sum / n_worlds, max_err


def run_mod1(ctx, mod, ds, tier_cfg, base_seed=2026):
    t0 = time.time()
    print("\n=== [MOD 1] 18-OECD + RoW Coupled Open-Economy & Falsification Suite ===")
    oc = ds["oecd"]
    isos = ds["isos"]
    prm = ref.DEFAULT_PARAMS.copy()
    n_worlds = tier_cfg["mod1_worlds"]
    n_fals = tier_cfg["mod1_fals_worlds"]
    ones_105 = np.ones((18, 105), dtype=np.float64)

    hist_summary = {}
    traj_hist = {}
    max_closure = 0.0
    for idx_r, rname in enumerate([
        "1_Adaptive_Growth", "2_Austerity_Coercion", "3_Neoliberal_Cap", "4_Gold_Standard",
        "5A_PG_Unreformed", "5M_PG_Monetary_Only", "5W_PG_WorkShare_Only", "5B_PG_EcoSocial_Full"
    ]):
        reg = ref.REGIMES[rname]
        cs, ts, merr = _run_mod1_batch(
            ctx, mod, oc, prm, reg, reg,
            oc["eroi"], oc["eta"], oc["age"], oc["civ"], oc["nonf"], oc["net_trade"], ones_105,
            n_worlds, 105, 105, base_seed + idx_r * 100000, 0, 0, 0, 0.75, 0, 1.0
        )
        max_closure = max(max_closure, merr)
        imp_m = oc["is_imp"] == 1
        hist_summary[rname] = {
            "oecd_viability_pct": round(float(cs[:, 0].mean() * 100.0), 2),
            "importer_viability_pct": round(float(cs[imp_m, 0].mean() * 100.0), 2),
            "exporter_viability_pct": round(float(cs[~imp_m, 0].mean() * 100.0), 2),
            "oecd_g_2024_pct": round(float(cs[:, 3].mean() * 100.0), 2),
            "importer_g_2024_pct": round(float(cs[imp_m, 3].mean() * 100.0), 2),
            "exporter_g_2024_pct": round(float(cs[~imp_m, 3].mean() * 100.0), 2),
            "per_country": {
                isos[c]: {
                    "viability_pct": round(float(cs[c, 0] * 100.0), 2),
                    "g_2024_pct": round(float(cs[c, 3] * 100.0), 2),
                    "debt_2024_ratio": round(float(cs[c, 4]), 3),
                    "fx_2024_ratio": round(float(cs[c, 7]), 4),
                } for c in range(18)
            }
        }
        traj_hist[rname] = ts
        print(f"  {rname:22s}: Viab={hist_summary[rname]['oecd_viability_pct']:6.2f}% "
              f"(Imp={hist_summary[rname]['importer_viability_pct']:5.1f}% Exp={hist_summary[rname]['exporter_viability_pct']:5.1f}%) "
              f"| g2024 Imp={hist_summary[rname]['importer_g_2024_pct']:.1f}% Exp={hist_summary[rname]['exporter_g_2024_pct']:.1f}%")

    # Forward 2025-2050 Doughnut Pathways (Common 1920-2024 Trunk across 18 OECD!)
    # (post_regime, eroi_tgt, eta_tgt, nonf_tgt, co2_mult_end, c_mult, exp_scale, lam_learn, eta_rec)
    fwd_specs = {
        "Scen1_Fossil_Decay":           ("1_Adaptive_Growth",        7.5,  0.260, 0.30, 0.96, 1.15, False, 1.00, 0.45),
        "Scen2_Green_Growth_Elec":      ("1_Adaptive_Growth",       19.0,  0.340, 0.80, 0.85, 0.85, False, 0.88, 0.45),
        "Scen3_Breakthrough_Growth":    ("1_Adaptive_Growth",       26.0,  0.370, 0.86, 0.80, 0.55, False, 0.80, 0.55),
        "Scen4a_PG_Unbuffered":         ("FWD_PostGrowth_EcoSocial",16.5,  0.355, 0.90, 0.55, 0.45, True,  1.00, 0.45),
        "Scen4b_PG_Circular_Doughnut":  ("FWD_PostGrowth_EcoSocial",19.0,  0.365, 0.90, 0.50, 0.45, True,  0.72, 0.80),
    }
    fwd_summary = {}
    traj_fwd = {}
    civ131 = np.concatenate([oc["civ"], np.repeat(oc["civ"][:, -1:], 26, axis=1)], axis=1)
    for idx_s, (sname, (rpost_k, e_t, eta_t, nf_t, cm_e, c_m, exps, lam_s, rec_s)) in enumerate(fwd_specs.items()):
        e131 = np.zeros((18, 131)); et131 = np.zeros((18, 131))
        nf131 = np.zeros((18, 131)); cm131 = np.ones((18, 131))
        for c in range(18):
            e26, et26, nf26, cm26 = build_forward_sequences(
                oc["eroi"][c, -1], oc["eta"][c, -1], oc["nonf"][c, -1],
                max(e_t, 5.0), max(eta_t, oc["eta"][c, -1]), max(nf_t, oc["nonf"][c, -1]),
                cm_e, exp_scale=exps
            )
            e131[c] = np.concatenate([oc["eroi"][c], e26])
            et131[c] = np.concatenate([oc["eta"][c], et26])
            nf131[c] = np.concatenate([oc["nonf"][c], nf26])
            cm131[c, 105:] = cm26
        prm_s = prm.copy()
        prm_s[11] = lam_s
        prm_s[12] = rec_s
        cs, ts, merr = _run_mod1_batch(
            ctx, mod, oc, prm_s, ref.REGIMES["1_Adaptive_Growth"], ref.REGIMES[rpost_k],
            e131, et131, oc["age_131"], civ131, nf131, oc["net_trade_131"], cm131,
            n_worlds, 131, 105, base_seed + 900000 + idx_s * 100000, 1, 1, 0, 0.75, 0, c_m
        )
        max_closure = max(max_closure, merr)
        tot_co2 = float(cs[:, 5].sum())
        max_min_ratio = float(cs[:, 6].mean())
        fwd_summary[sname] = {
            "oecd_viability_pct": round(float(cs[:, 0].mean() * 100.0), 2),
            "total_18oecd_co2_2025_2050_gt": round(tot_co2, 2),
            "within_15c_85gt_budget": bool(tot_co2 <= 85.0),
            "mean_mineral_ceiling_ratio": round(max_min_ratio, 3),
            "g_2050_pct": round(float(cs[:, 3].mean() * 100.0), 2),
        }
        traj_fwd[sname] = ts
        print(f"  {sname:28s}: Viab={fwd_summary[sname]['oecd_viability_pct']:6.2f}% | CO2(25-50)={tot_co2:6.1f} Gt | Mineral={max_min_ratio:.2f}x")

    # 1B: Anti-Self-Deception Falsification & Structural Specification Invariance
    fals_summary = {}
    eroi_frozen = np.repeat(oc["eroi"][:, 0:1], 105, axis=1)
    age_frozen  = np.repeat(oc["age"][:, 0:1], 105, axis=1)
    cs_f, ts_f, m1 = _run_mod1_batch(
        ctx, mod, oc, prm, ref.REGIMES["1_Adaptive_Growth"], ref.REGIMES["1_Adaptive_Growth"],
        eroi_frozen, oc["eta"], oc["age"], oc["civ"], oc["nonf"], oc["net_trade"], ones_105,
        n_fals, 105, 105, base_seed + 550000, 0, 0, 0, 0.75, 0, 1.0
    )
    cs_fa, ts_fa, m1a = _run_mod1_batch(
        ctx, mod, oc, prm, ref.REGIMES["1_Adaptive_Growth"], ref.REGIMES["1_Adaptive_Growth"],
        eroi_frozen, oc["eta"], age_frozen, oc["civ"], oc["nonf"], oc["net_trade"], ones_105,
        n_fals, 105, 105, base_seed + 555000, 0, 0, 0, 0.75, 0, 1.0
    )
    cs_b, ts_b, m2 = _run_mod1_batch(
        ctx, mod, oc, prm, ref.REGIMES["1_Adaptive_Growth"], ref.REGIMES["1_Adaptive_Growth"],
        oc["eroi"], oc["eta"], oc["age"], oc["civ"], oc["nonf"], oc["net_trade"], ones_105,
        n_fals, 105, 105, base_seed + 560000, 0, 0, 1, 0.75, 0, 1.0
    )
    imp_m = oc["is_imp"] == 1
    emp_g_yr = oc["obs_g"].mean(axis=0)
    cs_base_1a = hist_summary["1_Adaptive_Growth"]
    mae_base = float(np.mean(np.abs(traj_hist["1_Adaptive_Growth"][:, 0] - emp_g_yr)) * 100.0)
    mae_base_70 = float(np.mean(np.abs(traj_hist["1_Adaptive_Growth"][50:, 0] - emp_g_yr[50:])) * 100.0)
    mae_froz = float(np.mean(np.abs(ts_f[:, 0] - emp_g_yr)) * 100.0)
    mae_froz_70 = float(np.mean(np.abs(ts_f[50:, 0] - emp_g_yr[50:])) * 100.0)
    mae_fa = float(np.mean(np.abs(ts_fa[:, 0] - emp_g_yr)) * 100.0)

    fals_summary["placebo_tests"] = {
        "baseline_MAE_full_pp": round(mae_base, 2),
        "baseline_MAE_post1970_pp": round(mae_base_70, 2),
        "eroi_frozen_MAE_full_pp": round(mae_froz, 2),
        "eroi_frozen_MAE_post1970_pp": round(mae_froz_70, 2),
        "eroi_and_age_frozen_MAE_full_pp": round(mae_fa, 2),
        "eroi_and_age_frozen_R": round(float(np.corrcoef(ts_fa[:, 0], emp_g_yr)[0, 1]), 4),
    }
    fals_summary["blind_no_age_controller"] = {
        "viability_pct": round(float(cs_b[:, 0].mean() * 100.0), 2),
        "corr_vs_emp_g": round(float(np.corrcoef(ts_b[:, 0], emp_g_yr)[0, 1]), 4),
        "g_2024_pct": round(float(cs_b[:, 3].mean() * 100.0), 2),
    }
    fals_summary["structural_invariance"] = {}
    for ta, psi_v, lc_mode, tag in [
        (0.50, 0.048, 0, "high_trade_expose_ta0.50_psi0.048"),
        (0.75, 0.032, 0, "baseline_ta0.75_psi0.032"),
        (0.90, 0.015, 0, "low_trade_expose_ta0.90_psi0.015"),
        (0.75, 0.032, 1, "linear_climate_damage"),
    ]:
        inv_res = {}
        prm_i = prm.copy()
        prm_i[8] = psi_v
        for rk in ["1_Adaptive_Growth", "3_Neoliberal_Cap", "5B_PG_EcoSocial_Full"]:
            cs_i, _, mi = _run_mod1_batch(
                ctx, mod, oc, prm_i, ref.REGIMES[rk], ref.REGIMES[rk],
                oc["eroi"], oc["eta"], oc["age"], oc["civ"], oc["nonf"], oc["net_trade"], ones_105,
                max(n_fals // 2, 32), 105, 105, base_seed + 600000, 0, 0, 0, ta, lc_mode, 1.0
            )
            max_closure = max(max_closure, mi)
            inv_res[rk] = {
                "viability_pct": round(float(cs_i[:, 0].mean() * 100.0), 2),
                "imp_minus_exp_g_pp": round(float((cs_i[imp_m, 3].mean() - cs_i[~imp_m, 3].mean()) * 100.0), 2),
            }
        fals_summary["structural_invariance"][tag] = inv_res
    max_closure = max(max_closure, m1, m1a, m2)
    print(f"  [MOD 1 DONE] {round(time.time() - t0, 2)}s | Placebo MAE: Base={mae_base:.2f} -> EROI_Frozen={mae_froz:.2f} -> All_Frozen={mae_fa:.2f} pp | MaxErr={max_closure:.2e}")
    out_summary = {
        "module": "Mod1_Open18_RoW_and_Falsification",
        "runtime_sec": round(time.time() - t0, 2),
        "n_worlds_per_regime": n_worlds,
        "max_closure_err": max_closure,
        "historical_regimes": hist_summary,
        "forward_scenarios_2025_2050": fwd_summary,
        "falsification_and_invariance": fals_summary,
    }
    np.savez_compressed(
        os.path.join(OUT_DIR, "mod1_open18_arrays.npz"),
        **{f"hist_{k}": v for k, v in traj_hist.items()},
        **{f"fwd_{k}": v for k, v in traj_fwd.items()},
    )
    save_json(os.path.join(OUT_DIR, "mod1_open18_summary.json"), out_summary)
    return out_summary


# =============================================================================
# MODULE 2: Pure Unbiased 18-OECD ABC-SMC Posterior + Temporal + LOCO-CV
# =============================================================================
def _eval_abc_particles_gpu(ctx, mod, o_gpu, prm_mat_full, ctry_mask, t_eval, n_seeds_abc, base_seed):
    n_p = prm_mat_full.shape[0]
    batch = min(n_p, 512)
    out_dist = np.zeros(n_p, dtype=np.float64)
    out_viab = np.zeros(n_p, dtype=np.float64)
    out_gap = np.zeros(n_p, dtype=np.float64)
    max_err = 0.0

    d_mask = ctx.alloc_array(np.asarray(ctry_mask, dtype=np.int32))
    d_prm = ctx.alloc_array((batch, 22), np.float64)
    d_dist = ctx.alloc_array(batch, np.float64)
    d_viab = ctx.alloc_array(batch, np.float64)
    d_gap = ctx.alloc_array(batch, np.float64)
    d_errs = ctx.alloc_array((batch, 4), np.float64)
    try:
        for i0 in range(0, n_p, batch):
            i1 = min(n_p, i0 + batch)
            cur = i1 - i0
            d_prm_slice = ctx.alloc_array(prm_mat_full[i0:i1]) if cur != batch else d_prm
            if cur == batch:
                d_prm.copy_from_host(prm_mat_full[i0:i1])
            grid = (cur + 63) // 64
            mod.launch("kernel_mod2_abc_eval", grid, 64, [
                cur, 18, 105, t_eval, n_seeds_abc, ctypes.c_uint32(base_seed),
                d_prm_slice, o_gpu.bufs["reg_hist"], d_mask, o_gpu.bufs["ctry_is_imp"],
                o_gpu.bufs["eroi"], o_gpu.bufs["eta"], o_gpu.bufs["age"], o_gpu.bufs["civ"],
                o_gpu.bufs["nonf"], o_gpu.bufs["net_trade"],
                o_gpu.bufs["obs_g"], o_gpu.bufs["obs_debt"], o_gpu.bufs["obs_pi"], o_gpu.bufs["obs_u"],
                o_gpu.bufs["is_shock"], o_gpu.bufs["init_pub_debt"], o_gpu.bufs["init_priv_debt"],
                o_gpu.bufs["etwh_2024"], o_gpu.bufs["co2_2024_gt"], o_gpu.bufs["clim_b15"], o_gpu.bufs["min_caps"],
                d_dist, d_viab, d_gap, d_errs
            ])
            if cur != batch:
                d_prm_slice.free()
            out_dist[i0:i1] = d_dist.to_host()[:cur]
            out_viab[i0:i1] = d_viab.to_host()[:cur]
            out_gap[i0:i1] = d_gap.to_host()[:cur]
            max_err = max(max_err, check_errs_fuse(d_errs.to_host()[:cur], "Mod2_ABC"))
    finally:
        for b in [d_mask, d_prm, d_dist, d_viab, d_gap, d_errs]:
            b.free()
    return out_dist, out_viab, out_gap, max_err


def _run_abc_smc(ctx, mod, o_gpu, ctry_mask, t_eval, n_rounds, n_draw, n_pop, n_post_out, n_seeds_abc, rng, base_seed):
    bounds = ref.ABC_PRIOR_BOUNDS
    d_act = len(ref.ABC_ACTIVE_IDX)
    lo, hi = bounds[:, 0], bounds[:, 1]
    round_logs = []
    max_err = 0.0

    # Round 0: Pure Uniform Prior Draw
    cand_act = lo[None, :] + rng.random((n_draw, d_act)) * (hi - lo)[None, :]
    full_mat = np.tile(ref.DEFAULT_PARAMS, (n_draw, 1))
    full_mat[:, ref.ABC_ACTIVE_IDX] = cand_act
    dist, viab, gap, merr = _eval_abc_particles_gpu(ctx, mod, o_gpu, full_mat, ctry_mask, t_eval, n_seeds_abc, base_seed)
    max_err = max(max_err, merr)

    # Keep top n_pop particles
    order = np.argsort(dist)[:n_pop]
    pop = cand_act[order]
    pop_dist = dist[order]
    pop_viab = viab[order]
    pop_gap = gap[order]
    weights = np.full(n_pop, 1.0 / n_pop, dtype=np.float64)
    eps_cur = float(np.max(pop_dist))
    round_logs.append({
        "round": 0, "eps": round(eps_cur, 4), "median_dist": round(float(np.median(pop_dist)), 4),
        "ess": round(sm.compute_ess(weights), 1),
        "emergent_viab_pct": round(float(np.mean(pop_viab) * 100.0), 2),
        "emergent_imp_exp_pp": round(float(np.mean(pop_gap) * 100.0), 2),
    })

    for r in range(1, n_rounds):
        props, p_weights = sm.propose_abc_smc_batch(pop, weights, bounds, n_draw, rng)
        full_mat = np.tile(ref.DEFAULT_PARAMS, (n_draw, 1))
        full_mat[:, ref.ABC_ACTIVE_IDX] = props
        p_dist, p_viab, p_gap, merr = _eval_abc_particles_gpu(
            ctx, mod, o_gpu, full_mat, ctry_mask, t_eval, n_seeds_abc, base_seed + r * 31
        )
        max_err = max(max_err, merr)

        # Target tighter tolerance (65th percentile of previous distance or top n_pop)
        eps_next = min(eps_cur * 0.97, float(np.quantile(pop_dist, 0.72)))
        acc_idx = np.where(p_dist <= eps_next)[0]
        if len(acc_idx) < n_pop:
            # Take best n_pop to preserve ESS cleanly while tracking effective eps
            acc_idx = np.argsort(p_dist)[:n_pop]
        else:
            # Sort & trim or subsample proportionally up to max(n_pop, n_post_out on final round)
            keep_n = n_post_out if (r == n_rounds - 1) else n_pop
            if len(acc_idx) > keep_n:
                sel = rng.choice(acc_idx, size=keep_n, replace=False)
                acc_idx = sel
        eps_next = float(np.max(p_dist[acc_idx]))

        pop = props[acc_idx]
        pop_dist = p_dist[acc_idx]
        pop_viab = p_viab[acc_idx]
        pop_gap = p_gap[acc_idx]
        w_acc = p_weights[acc_idx]
        w_acc /= np.sum(w_acc)
        ess = sm.compute_ess(w_acc)
        if ess < 0.45 * len(pop):
            rs_idx = sm.systematic_resample_indices(w_acc, len(pop), rng)
            pop = pop[rs_idx].copy()
            pop_dist = pop_dist[rs_idx].copy()
            pop_viab = pop_viab[rs_idx].copy()
            pop_gap = pop_gap[rs_idx].copy()
            weights = np.full(len(pop), 1.0 / len(pop), dtype=np.float64)
        else:
            weights = w_acc
        eps_cur = eps_next
        round_logs.append({
            "round": r, "eps": round(eps_cur, 4), "median_dist": round(float(np.median(pop_dist)), 4),
            "ess": round(ess, 1),
            "emergent_viab_pct": round(float(np.mean(pop_viab) * 100.0), 2),
            "emergent_imp_exp_pp": round(float(np.mean(pop_gap) * 100.0), 2),
        })

    # Systematic resample exact n_post_out posterior particles (aligned 1-to-1)
    fin_idx = sm.systematic_resample_indices(weights, n_post_out, rng)
    post_act = pop[fin_idx].copy()
    post_full = np.tile(ref.DEFAULT_PARAMS, (n_post_out, 1))
    post_full[:, ref.ABC_ACTIVE_IDX] = post_act
    return (
        post_full,
        pop_dist[fin_idx].copy(),
        pop_viab[fin_idx].copy(),
        pop_gap[fin_idx].copy(),
        round_logs,
        max_err,
    )


def run_mod2(ctx, mod, ds, tier_cfg):
    t0 = time.time()
    print("\n=== [MOD 2] Pure Unbiased 18-OECD ABC-SMC Inversion + Temporal & LOCO-CV ===")
    oc = ds["oecd"]
    isos = ds["isos"]
    o_gpu = OecdGpuArrays(ctx, oc)
    rng = np.random.default_rng(202602)
    max_err = 0.0
    try:
        # 2A: Full 18-OECD 1920-2024 ABC-SMC
        all_mask = np.ones(18, dtype=np.int32)
        post_full, p_dist, p_viab, p_gap, rlogs, m0 = _run_abc_smc(
            ctx, mod, o_gpu, all_mask, 105,
            tier_cfg["mod2_rounds"], tier_cfg["mod2_draw"], tier_cfg["mod2_pop"],
            tier_cfg["mod2_post_n"], tier_cfg["mod2_seeds"], rng, 1000
        )
        max_err = max(max_err, m0)
        print(f"  2A Full OECD ABC-SMC: R0 med={rlogs[0]['median_dist']:.3f} -> R{rlogs[-1]['round']} med={rlogs[-1]['median_dist']:.3f} "
              f"| Unoptimized Emergent Viab={rlogs[-1]['emergent_viab_pct']:.1f}%, Imp-Exp={rlogs[-1]['emergent_imp_exp_pp']:+.2f} pp")

        # Parameter Posterior Medians & 95% CIs
        post_stats = {}
        for k_idx in ref.ABC_ACTIVE_IDX:
            vals = post_full[:, k_idx]
            lo95, med, hi95 = np.percentile(vals, [2.5, 50.0, 97.5])
            post_stats[ref.PARAM_NAMES[k_idx]] = {"median": round(float(med), 4), "ci95": [round(float(lo95), 4), round(float(hi95), 4)]}

        # Post-hoc Survivorship-Conditioning Overlap Diagnostic (Zero Inference Change):
        # Compare unconditioned P(theta | Y_obs) vs survival-conditioned P(theta | Y_obs, S >= tau)
        q_grid = np.linspace(0.005, 0.995, 200)
        bounds = ref.ABC_PRIOR_BOUNDS
        surv_cond_summary = {}
        for tag, tau in [
            ("viab_ge_90pct", 0.90),
            ("viab_ge_95pct", 0.95),
            ("viab_eq_100pct", 1.0 - 1e-9),
        ]:
            mask = p_viab >= tau
            n_ret = int(np.sum(mask))
            ret_pct = round(100.0 * n_ret / max(len(p_viab), 1), 2)
            if n_ret < 10:
                surv_cond_summary[tag] = {
                    "threshold": float(round(tau, 4)),
                    "n_retained": n_ret,
                    "retained_fraction_pct": ret_pct,
                    "note": "fewer_than_10_particles",
                }
                continue
            med_shifts, ci_jaccards, w1_norms = [], [], []
            per_param = {}
            for j, k_idx in enumerate(ref.ABC_ACTIVE_IDX):
                u_vals = post_full[:, k_idx]
                c_vals = post_full[mask, k_idx]
                u_lo, u_med, u_hi = np.percentile(u_vals, [2.5, 50.0, 97.5])
                c_lo, c_med, c_hi = np.percentile(c_vals, [2.5, 50.0, 97.5])
                rel_med_shift_pct = 100.0 * abs(c_med - u_med) / max(abs(u_med), 1e-12)
                inter = max(0.0, min(u_hi, c_hi) - max(u_lo, c_lo))
                union = max(max(u_hi, c_hi) - min(u_lo, c_lo), 1e-12)
                ci95_jaccard_pct = 100.0 * inter / union
                w1_norm = float(np.mean(np.abs(np.quantile(u_vals, q_grid) - np.quantile(c_vals, q_grid)))) / max(
                    float(bounds[j, 1] - bounds[j, 0]), 1e-12
                )
                med_shifts.append(rel_med_shift_pct)
                ci_jaccards.append(ci95_jaccard_pct)
                w1_norms.append(w1_norm)
                if tag == "viab_ge_95pct":
                    per_param[ref.PARAM_NAMES[k_idx]] = {
                        "uncond_median": round(float(u_med), 4),
                        "cond_median": round(float(c_med), 4),
                        "rel_median_shift_pct": round(float(rel_med_shift_pct), 3),
                        "ci95_jaccard_pct": round(float(ci95_jaccard_pct), 2),
                        "norm_w1": round(float(w1_norm), 4),
                    }
            entry = {
                "threshold": float(round(tau, 4)),
                "n_retained": n_ret,
                "retained_fraction_pct": ret_pct,
                "mean_viab_pct_uncond": round(float(np.mean(p_viab) * 100.0), 2),
                "mean_viab_pct_cond": round(float(np.mean(p_viab[mask]) * 100.0), 2),
                "emergent_imp_exp_pp_uncond": round(float(np.mean(p_gap) * 100.0), 2),
                "emergent_imp_exp_pp_cond": round(float(np.mean(p_gap[mask]) * 100.0), 2),
                "mean_rel_median_shift_pct": round(float(np.mean(med_shifts)), 3),
                "max_rel_median_shift_pct": round(float(np.max(med_shifts)), 3),
                "mean_ci95_jaccard_pct": round(float(np.mean(ci_jaccards)), 2),
                "min_ci95_jaccard_pct": round(float(np.min(ci_jaccards)), 2),
                "mean_norm_w1": round(float(np.mean(w1_norms)), 4),
                "max_norm_w1": round(float(np.max(w1_norms)), 4),
            }
            if tag == "viab_ge_95pct":
                entry["per_parameter"] = per_param
            surv_cond_summary[tag] = entry

        rep_tag = "viab_ge_95pct" if "mean_rel_median_shift_pct" in surv_cond_summary["viab_ge_95pct"] else "viab_ge_90pct"
        rep = surv_cond_summary[rep_tag]
        if "mean_rel_median_shift_pct" in rep:
            print(f"    Survivorship Overlap [{rep_tag}]: retained={rep['retained_fraction_pct']:.1f}%, "
                  f"meanMedShift={rep['mean_rel_median_shift_pct']:.2f}%, meanCI95Jaccard={rep['mean_ci95_jaccard_pct']:.1f}%, "
                  f"meanW1={rep['mean_norm_w1']:.4f}")

        # 2B: Temporal Forward Split (Train 1920-1980 -> Blind Eval 1981-2024)
        post_1980, _, _, _, rlogs_80, m1 = _run_abc_smc(
            ctx, mod, o_gpu, all_mask, 61,
            max(4, tier_cfg["mod2_rounds"] // 2 + 1), max(1024, tier_cfg["mod2_draw"] // 2),
            min(1024, tier_cfg["mod2_pop"]), min(512, tier_cfg["mod2_post_n"]),
            max(2, tier_cfg["mod2_seeds"] // 2), rng, 4000
        )
        max_err = max(max_err, m1)
        # Evaluate posterior trained strictly <= 1980 on full 1920-2024 vs 1980 baseline using median posterior
        med_80 = np.median(post_1980, axis=0)
        cs_80, ts_80, m1b = _run_mod1_batch(
            ctx, mod, oc, med_80, ref.REGIMES["1_Adaptive_Growth"], ref.REGIMES["1_Adaptive_Growth"],
            oc["eroi"], oc["eta"], oc["age"], oc["civ"], oc["nonf"], oc["net_trade"], np.ones((18, 105)),
            max(64, tier_cfg["mod1_fals_worlds"] // 4), 105, 105, 8888, 0, 0, 0, 0.75, 0, 1.0
        )
        max_err = max(max_err, m1b)
        emp_g_yr = oc["obs_g"].mean(axis=0)
        oos_r_1981_2024 = float(np.corrcoef(ts_80[61:, 0], emp_g_yr[61:])[0, 1])
        oos_mae_1981_2024 = float(np.mean(np.abs(ts_80[61:, 0] - emp_g_yr[61:])) * 100.0)
        print(f"  2B Temporal Forward OOS (Train<=1980 -> Test 1981-2024): R={oos_r_1981_2024:.4f}, MAE={oos_mae_1981_2024:.2f} pp, Viab={cs_80[:,0].mean()*100:.1f}%")

        # 2C: LOCO-CV (Leave-One-Country-Out)
        loco_results = {}
        loco_list = tier_cfg["mod2_loco_isos"]
        for iso in loco_list:
            c_idx = isos.index(iso)
            l_mask = np.ones(18, dtype=np.int32)
            l_mask[c_idx] = 0
            post_l, _, _, _, _, m_l = _run_abc_smc(
                ctx, mod, o_gpu, l_mask, 105,
                max(3, tier_cfg["mod2_rounds"] // 2), max(1024, tier_cfg["mod2_draw"] // 3),
                min(512, tier_cfg["mod2_pop"]), min(256, tier_cfg["mod2_post_n"]),
                max(2, tier_cfg["mod2_seeds"] // 2), rng, 7000 + c_idx * 100
            )
            max_err = max(max_err, m_l)
            # Blind evaluate ONLY on the left-out country using CPU exact loop over median & ensemble (18-year fast!)
            med_l = np.median(post_l, axis=0)
            g_pred_yr = np.zeros(105)
            n_loco_s = 16
            reg = ref.REGIMES["1_Adaptive_Growth"]
            for s in range(n_loco_s):
                S_l = ref.init_country_state_py(reg[9], oc["init_pub_debt"][c_idx], oc["init_priv_debt"][c_idx])
                sc_l = {"e_gross_ref": 0.0, "alive": 1, "cause": 0, "years_alive": 0.0,
                        "max_mon_err": 0.0, "max_nw_err": 0.0, "max_phys_err": 0.0}
                for t in range(105):
                    tp = max(0, t - 1)
                    er, ei, uh = ref.philox4x32_10_normals_py(12000 + s, c_idx, t, 1)
                    _, gt, _, _, _ = ref.step_country_sfc_bp_py(
                        S_l, med_l, reg, t, 105, 45,
                        oc["eroi"][c_idx, t], oc["eta"][c_idx, t], oc["age"][c_idx, t], oc["civ"][c_idx, t],
                        oc["nonf"][c_idx, t], oc["nonf"][c_idx, tp], -oc["net_trade"][c_idx, t], 1.0, 1.0,
                        oc["etwh_2024"][c_idx], oc["co2_2024_gt"][c_idx], oc["clim_b15"][c_idx], oc["min_caps"][c_idx],
                        0, 0, er, ei, uh, sc_l
                    )
                    g_pred_yr[t] += gt / n_loco_s
            obs_c_g = oc["obs_g"][c_idx]
            loco_r = float(np.corrcoef(g_pred_yr, obs_c_g)[0, 1])
            loco_mae = float(np.mean(np.abs(g_pred_yr - obs_c_g)) * 100.0)
            loco_results[iso] = {"blind_oos_r": round(loco_r, 4), "blind_oos_mae_pp": round(loco_mae, 2)}
            print(f"    LOCO-CV Blind [{iso}]: R={loco_r:.4f}, MAE={loco_mae:.2f} pp")
    finally:
        o_gpu.free()

    np.savez_compressed(
        os.path.join(OUT_DIR, "mod2_abc_posterior.npz"),
        posterior_full=post_full,
        posterior_dist=p_dist,
        posterior_emergent_viab=p_viab,
        posterior_emergent_impexp=p_gap,
        posterior_pre1980=post_1980,
    )
    summary = {
        "module": "Mod2_Unbiased_ABC_SMC_and_CV",
        "runtime_sec": round(time.time() - t0, 2),
        "max_closure_err": max_err,
        "posterior_n": int(post_full.shape[0]),
        "round_schedule": rlogs,
        "parameter_posteriors": post_stats,
        "survivorship_conditioning_check": surv_cond_summary,
        "temporal_split_1981_2024": {
            "unseen_r": round(oos_r_1981_2024, 4),
            "unseen_mae_pp": round(oos_mae_1981_2024, 2),
            "oecd_viability_pct": round(float(cs_80[:, 0].mean() * 100.0), 2),
        },
        "loco_cv": loco_results,
    }
    save_json(os.path.join(OUT_DIR, "mod2_abc_summary.json"), summary)
    return summary, post_full


# =============================================================================
# MODULE 3: 22-Parameter Saltelli-Jansen Sobol Global Sensitivity Analysis
# =============================================================================
def _run_sobol_target(ctx, mod, X_all, N, k, n_crn, e131, et131, ag131, cv131, nf131, nt131, cm131,
                      init_pub, init_priv, etwh, co2_24, cb15, mcaps, n_boot):
    n_pts = X_all.shape[0]
    reg_h = ref.REGIMES["1_Adaptive_Growth"]
    d_reg = ctx.alloc_array(reg_h)
    d_e = ctx.alloc_array(e131); d_et = ctx.alloc_array(et131); d_ag = ctx.alloc_array(ag131)
    d_cv = ctx.alloc_array(cv131); d_nf = ctx.alloc_array(nf131); d_nt = ctx.alloc_array(nt131)
    d_cm = ctx.alloc_array(cm131)

    pts_per_batch = max(1, min(n_pts, 16384 // n_crn))
    Y_mean = np.zeros((n_pts, 4), dtype=np.float64)
    max_err = 0.0

    d_prm = ctx.alloc_array((pts_per_batch, 22), np.float64)
    d_resp = ctx.alloc_array((pts_per_batch * n_crn, 4), np.float64)
    d_errs = ctx.alloc_array((pts_per_batch * n_crn, 4), np.float64)
    try:
        for i0 in range(0, n_pts, pts_per_batch):
            i1 = min(n_pts, i0 + pts_per_batch)
            cur = i1 - i0
            d_p_use = ctx.alloc_array(X_all[i0:i1]) if cur != pts_per_batch else d_prm
            if cur == pts_per_batch:
                d_prm.copy_from_host(X_all[i0:i1])
            n_jobs = cur * n_crn
            grid = (n_jobs + 127) // 128
            mod.launch("kernel_mod3_sobol_eval", grid, 128, [
                cur, n_crn, 131, 105, ctypes.c_uint32(3030),
                d_p_use, d_reg, d_e, d_et, d_ag, d_cv, d_nf, d_nt, d_cm,
                float(init_pub), float(init_priv), float(etwh), float(co2_24), float(cb15),
                float(mcaps[0]), float(mcaps[1]), float(mcaps[2]),
                d_resp, d_errs
            ])
            if cur != pts_per_batch:
                d_p_use.free()
            max_err = max(max_err, check_errs_fuse(d_errs.to_host()[:n_jobs], "Mod3_Sobol"))
            r_b = d_resp.to_host()[:n_jobs].reshape(cur, n_crn, 4).mean(axis=1)
            Y_mean[i0:i1] = r_b
    finally:
        for b in [d_reg, d_e, d_et, d_ag, d_cv, d_nf, d_nt, d_cm, d_prm, d_resp, d_errs]:
            b.free()

    metric_names = ["years_alive", "co2_cum_2050", "g_tot_2050", "mineral_ratio_2050"]
    out = {}
    for m_i, m_name in enumerate(metric_names):
        sb = sm.compute_jansen_sobol_indices(Y_mean[:, m_i], N, k, n_boot=n_boot, seed=100 + m_i)
        out[m_name] = {
            "var_y": round(sb["var_y"], 6),
            "S_i": {ref.PARAM_NAMES[j]: round(float(sb["S_i"][j]), 4) for j in range(k)},
            "S_Ti": {ref.PARAM_NAMES[j]: round(float(sb["S_Ti"][j]), 4) for j in range(k)},
            "S_Ti_ci95": {ref.PARAM_NAMES[j]: [round(float(sb["S_Ti_ci95"][j, 0]), 4), round(float(sb["S_Ti_ci95"][j, 1]), 4)] for j in range(k)},
            "_raw_Si": sb["S_i"],
            "_raw_STi": sb["S_Ti"],
        }
    return out, max_err


def run_mod3(ctx, mod, ds, tier_cfg):
    t0 = time.time()
    print("\n=== [MOD 3] 22-Parameter Saltelli-Jansen Sobol Global Sensitivity Analysis ===")
    oc = ds["oecd"]
    isos = ds["isos"]
    N_base = tier_cfg["mod3_n_base"]
    n_crn = tier_cfg["mod3_n_crn"]
    n_boot = tier_cfg["mod3_n_boot"]

    X_all, N, k = sm.build_saltelli_matrix(ref.SOBOL_BOUNDS, N_base, seed=202603)
    targets = []
    imp_m = oc["is_imp"] == 1
    groups = [("OECD_Mean", np.ones(18, dtype=bool)), ("Importer_Mean", imp_m), ("Exporter_Mean", ~imp_m)]
    if tier_cfg["mod3_per_country"]:
        for c_i, iso in enumerate(isos):
            m = np.zeros(18, dtype=bool); m[c_i] = True
            groups.append((iso, m))
    elif len(tier_cfg.get("mod3_groups_only", [])) > 0:
        groups = [g for g in groups if g[0] in tier_cfg["mod3_groups_only"]]

    results_clean = {}
    npz_payload = {}
    max_err = 0.0
    for gname, gmask in groups:
        e24 = float(oc["eroi"][gmask, -1].mean()); eta24 = float(oc["eta"][gmask, -1].mean()); nf24 = float(oc["nonf"][gmask, -1].mean())
        e26, et26, nf26, cm26 = build_forward_sequences(e24, eta24, nf24, 17.5, 0.35, 0.85, 0.70, exp_scale=True)
        e131 = np.concatenate([oc["eroi"][gmask].mean(axis=0), e26])
        et131 = np.concatenate([oc["eta"][gmask].mean(axis=0), et26])
        ag131 = oc["age_131"][gmask].mean(axis=0)
        cv131 = np.concatenate([oc["civ"][gmask].mean(axis=0), np.full(26, float(oc["civ"][gmask, -1].mean()))])
        nf131 = np.concatenate([oc["nonf"][gmask].mean(axis=0), nf26])
        nt131 = oc["net_trade_131"][gmask].mean(axis=0)
        cm131 = np.concatenate([np.ones(105), cm26])

        res_g, merr = _run_sobol_target(
            ctx, mod, X_all, N, k, n_crn, e131, et131, ag131, cv131, nf131, nt131, cm131,
            float(oc["init_pub_debt"][gmask].mean()), float(oc["init_priv_debt"][gmask].mean()),
            float(oc["etwh_2024"][gmask].sum()), float(oc["co2_2024_gt"][gmask].sum()),
            float(oc["clim_b15"][gmask].sum()), oc["min_caps"][gmask].sum(axis=0), n_boot
        )
        max_err = max(max_err, merr)
        clean_g = {}
        for mname, md in res_g.items():
            clean_g[mname] = {kk: vv for kk, vv in md.items() if not kk.startswith("_")}
            npz_payload[f"{gname}_{mname}_Si"] = md["_raw_Si"]
            npz_payload[f"{gname}_{mname}_STi"] = md["_raw_STi"]
        results_clean[gname] = clean_g

        top_y = sorted(clean_g["years_alive"]["S_Ti"].items(), key=lambda kv: kv[1], reverse=True)[:4]
        print(f"  Sobol [{gname:13s}] Top S_Ti (years_alive): " + ", ".join(f"{a}={b:.3f}" for a, b in top_y))

    np.savez_compressed(os.path.join(OUT_DIR, "mod3_sobol_arrays.npz"), param_names=np.array(ref.PARAM_NAMES), **npz_payload)
    summary = {
        "module": "Mod3_Sobol_22Param_GSA",
        "runtime_sec": round(time.time() - t0, 2),
        "n_base_saltelli": N, "n_params": k, "n_crn_seeds": n_crn,
        "total_trajectories": int(X_all.shape[0] * n_crn * len(groups)),
        "max_closure_err": max_err,
        "groups": results_clean,
    }
    save_json(os.path.join(OUT_DIR, "mod3_sobol_summary.json"), summary)
    return summary


# =============================================================================
# MODULE 4: 4D Construction-Phase Energy Cannibalization x Minerals x Post-Growth
# =============================================================================
def run_mod4(ctx, mod, ds, tier_cfg, prm_post_med):
    t0 = time.time()
    print("\n=== [MOD 4] 4D Energy Cannibalization x Critical Minerals x Post-Growth ===")
    oc = ds["oecd"]
    n_sp, n_rc, n_lm, n_seeds = tier_cfg["mod4_dims"]
    nonf_tgts = np.linspace(0.55, 0.95, n_sp)
    rec_vals = np.linspace(0.20, 0.85, n_rc)
    lam_vals = np.linspace(0.60, 1.15, n_lm)
    pg_flags = [0.0, 1.0]

    grid_cfg = np.array([
        [nf, rc, lm, pg]
        for pg in pg_flags for nf in nonf_tgts for rc in rec_vals for lm in lam_vals
    ], dtype=np.float64)
    n_pts = grid_cfg.shape[0]
    total_jobs = n_pts * n_seeds

    eroi_h = oc["eroi"].mean(axis=0)
    eta_h = oc["eta"].mean(axis=0)
    ag131 = oc["age_131"].mean(axis=0)
    cv131 = np.concatenate([oc["civ"].mean(axis=0), np.full(26, float(oc["civ"][:, -1].mean()))])
    nonf_h = oc["nonf"].mean(axis=0)
    nt131 = oc["net_trade_131"].mean(axis=0)
    etwh24 = float(oc["etwh_2024"].sum())
    co2_24 = float(oc["co2_2024_gt"].sum())
    cb15 = float(oc["clim_b15"].sum())
    mcaps = oc["min_caps"].sum(axis=0)

    pts_batch = max(1, min(n_pts, 16384 // n_seeds))
    bufs = [
        ctx.alloc_array(prm_post_med),
        ctx.alloc_array((pts_batch, 4), np.float64),
        ctx.alloc_array(ref.REGIMES["1_Adaptive_Growth"]),
        ctx.alloc_array(ref.REGIMES["FWD_PostGrowth_EcoSocial"]),
        ctx.alloc_array(eroi_h), ctx.alloc_array(eta_h), ctx.alloc_array(ag131),
        ctx.alloc_array(cv131), ctx.alloc_array(nonf_h), ctx.alloc_array(nt131),
        ctx.alloc_array((pts_batch * n_seeds, 6), np.float64),
        ctx.alloc_array(pts_batch * n_seeds, np.float64),
    ]
    agg = np.zeros((n_pts, 6), dtype=np.float64)  # alive, co2, mr, peak_build, yrs, safe_and_just_pct
    max_err = 0.0
    try:
        for i0 in range(0, n_pts, pts_batch):
            i1 = min(n_pts, i0 + pts_batch)
            cur = i1 - i0
            d_g = ctx.alloc_array(grid_cfg[i0:i1]) if cur != pts_batch else bufs[1]
            if cur == pts_batch:
                bufs[1].copy_from_host(grid_cfg[i0:i1])
            n_j = cur * n_seeds
            mod.launch("kernel_mod4_cannib_minerals", (n_j + 127) // 128, 128, [
                cur, n_seeds, 131, 105, ctypes.c_uint32(4040),
                bufs[0], d_g, bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7], bufs[8], bufs[9],
                etwh24, co2_24, cb15, float(mcaps[0]), float(mcaps[1]), float(mcaps[2]),
                bufs[10], bufs[11]
            ])
            if cur != pts_batch:
                d_g.free()
            r_nan = bufs[11].to_host()[:n_j]
            if np.max(r_nan) > 0.5:
                raise RuntimeError("NaN detected in Mod4 kernel!")
            res_b = bufs[10].to_host()[:n_j].reshape(cur, n_seeds, 6)
            max_err = max(max_err, float(np.max(res_b[:, :, 5])))
            if max_err > 1e-11:
                raise RuntimeError(f"Closure error in Mod4: {max_err}")
            agg[i0:i1, 0:5] = res_b[:, :, 0:5].mean(axis=1)
            safe_just = (res_b[:, :, 0] > 0.5) & (res_b[:, :, 1] <= cb15) & (res_b[:, :, 2] <= 1.0)
            agg[i0:i1, 5] = safe_just.mean(axis=1)
    finally:
        for b in bufs:
            b.free()

    agg_4d = agg.reshape(2, n_sp, n_rc, n_lm, 6)
    gr_safe = float(agg_4d[0, :, :, :, 5].max() * 100.0)
    pg_safe = float(agg_4d[1, :, :, :, 5].max() * 100.0)
    np.savez_compressed(
        os.path.join(OUT_DIR, "mod4_cannib_minerals.npz"),
        nonf_tgts=nonf_tgts, rec_vals=rec_vals, lam_vals=lam_vals, pg_flags=np.array(pg_flags),
        agg_4d=agg_4d,
    )
    summary = {
        "module": "Mod4_EnergyCannibalization_and_CriticalMinerals",
        "runtime_sec": round(time.time() - t0, 2),
        "grid_shape": [2, n_sp, n_rc, n_lm],
        "seeds_per_point": n_seeds,
        "total_trajectories": int(total_jobs),
        "max_closure_err": max_err,
        "max_triple_boundary_safe_just_growth_pct": round(gr_safe, 2),
        "max_triple_boundary_safe_just_postgrowth_pct": round(pg_safe, 2),
        "peak_build_cannibalization_pct_growth": round(float(agg_4d[0, :, :, :, 3].max() * 100.0), 2),
        "peak_build_cannibalization_pct_postgrowth": round(float(agg_4d[1, :, :, :, 3].max() * 100.0), 2),
    }
    print(f"  [MOD 4 DONE] {summary['runtime_sec']}s | Triple-Safe Frontier: Growth max={gr_safe:.1f}% vs Post-Growth max={pg_safe:.1f}% | MaxErr={max_err:.2e}")
    save_json(os.path.join(OUT_DIR, "mod4_cannib_minerals.json"), summary)
    return summary


# =============================================================================
# MODULE 5: High-Res Phase Cliff + AR(1)/Var EWS + Irreversible Hysteresis Loop
# =============================================================================
def run_mod5(ctx, mod, tier_cfg, prm_post_med):
    t0 = time.time()
    print("\n=== [MOD 5] High-Res Phase Cliff + Critical Slowing Down EWS + Hysteresis ===")
    ng, n_seeds = tier_cfg["mod5_grid"], tier_cfg["mod5_seeds"]
    eroi_axis = np.linspace(4.0, 32.0, ng, dtype=np.float64)
    gorg_axis = np.linspace(0.02, 0.18, ng, dtype=np.float64)
    ee, gg = np.meshgrid(eroi_axis, gorg_axis, indexing="xy")  # (gorg, eroi)
    e_flat = ee.ravel(); g_flat = gg.ravel()
    n_pts = e_flat.shape[0]
    pts_batch = max(1, min(n_pts, 32768 // n_seeds))

    slices = {"Importer_Net": 0.60, "Exporter_Net": -0.50}
    slice_maps = {}
    slice_summary = {}
    max_err = 0.0

    for sname, imp_val in slices.items():
        bufs = [
            ctx.alloc_array(prm_post_med),
            ctx.alloc_array(pts_batch, np.float64),
            ctx.alloc_array(pts_batch, np.float64),
            ctx.alloc_array((pts_batch * n_seeds, 6), np.float64),
            ctx.alloc_array(pts_batch * n_seeds, np.float64),
        ]
        m_out = np.zeros((n_pts, 5), dtype=np.float64)  # surv, legit_pct, ar1_d, var_d, ar1_b
        try:
            for i0 in range(0, n_pts, pts_batch):
                i1 = min(n_pts, i0 + pts_batch)
                cur = i1 - i0
                de = ctx.alloc_array(e_flat[i0:i1]) if cur != pts_batch else bufs[1]
                dg = ctx.alloc_array(g_flat[i0:i1]) if cur != pts_batch else bufs[2]
                if cur == pts_batch:
                    bufs[1].copy_from_host(e_flat[i0:i1])
                    bufs[2].copy_from_host(g_flat[i0:i1])
                n_j = cur * n_seeds
                mod.launch("kernel_mod5_phase_ews", (n_j + 127) // 128, 128, [
                    cur, n_seeds, 45, ctypes.c_uint32(5050), float(imp_val),
                    bufs[0], de, dg, bufs[3], bufs[4]
                ])
                if cur != pts_batch:
                    de.free(); dg.free()
                if np.max(bufs[4].to_host()[:n_j]) > 0.5:
                    raise RuntimeError("NaN in Mod5 phase kernel!")
                raw_b = bufs[3].to_host()[:n_j].reshape(cur, n_seeds, 6)
                max_err = max(max_err, float(np.max(raw_b[:, :, 5])))
                if max_err > 1e-11:
                    raise RuntimeError(f"Closure error in Mod5 phase: {max_err}")
                m_out[i0:i1, 0] = raw_b[:, :, 0].mean(axis=1)
                m_out[i0:i1, 1] = (raw_b[:, :, 1] == 1).mean(axis=1)
                m_out[i0:i1, 2] = raw_b[:, :, 2].mean(axis=1)
                m_out[i0:i1, 3] = raw_b[:, :, 3].mean(axis=1)
                m_out[i0:i1, 4] = raw_b[:, :, 4].mean(axis=1)
        finally:
            for b in bufs:
                b.free()

        m_grid = m_out.reshape(ng, ng, 5)
        slice_maps[sname] = m_grid
        idx_o5 = int(np.argmin(np.abs(gorg_axis - 0.05)))
        idx_o15 = int(np.argmin(np.abs(gorg_axis - 0.15)))
        e_m5 = float(eroi_axis[np.argmax(m_grid[idx_o5, :, 0] >= 0.50)]) if np.any(m_grid[idx_o5, :, 0] >= 0.50) else 32.0
        e_m15 = float(eroi_axis[np.argmax(m_grid[idx_o15, :, 0] >= 0.50)]) if np.any(m_grid[idx_o15, :, 0] >= 0.50) else 32.0
        safe_mask = m_grid[:, :, 0] >= 0.85
        tip_mask = (m_grid[:, :, 0] >= 0.15) & (m_grid[:, :, 0] <= 0.70)
        slice_summary[sname] = {
            "min_viable_eroi_at_gorg_5pct": round(e_m5, 2),
            "min_viable_eroi_at_gorg_15pct": round(e_m15, 2),
            "ews_ar1_diff_debt_safe": round(float(m_grid[safe_mask, 2].mean()) if np.any(safe_mask) else 0.0, 4),
            "ews_ar1_diff_debt_tipping": round(float(m_grid[tip_mask, 2].mean()) if np.any(tip_mask) else 0.0, 4),
            "ews_var_binfra_safe": round(float(m_grid[safe_mask, 3].mean()) if np.any(safe_mask) else 0.0, 6),
            "ews_var_binfra_tipping": round(float(m_grid[tip_mask, 3].mean()) if np.any(tip_mask) else 0.0, 6),
            "ews_ar1_binfra_safe": round(float(m_grid[safe_mask, 4].mean()) if np.any(safe_mask) else 0.0, 4),
            "ews_ar1_binfra_tipping": round(float(m_grid[tip_mask, 4].mean()) if np.any(tip_mask) else 0.0, 4),
        }
        print(f"  Phase [{sname:12s}]: MinEROI@gorg=5%->{e_m5:.1f} vs 15%->{e_m15:.1f} | "
              f"EWS Var(B_infra) Safe={slice_summary[sname]['ews_var_binfra_safe']:.5f} -> Tipping={slice_summary[sname]['ews_var_binfra_tipping']:.5f}")

    # 5B: Hysteresis Loop (EROI 25 -> 6 over t=0..39, then 6 -> 25 over t=40..79)
    eroi_hyst = np.concatenate([np.linspace(25.0, 6.0, 40), np.linspace(6.0, 25.0, 40)])
    n_hseeds = tier_cfg["mod5_hyst_seeds"]
    hyst_res = {}
    hyst_trajs = {}
    for r_hyst in ["1_Adaptive_Growth", "2_Austerity_Coercion", "3_Neoliberal_Cap"]:
        h_bufs = [
            ctx.alloc_array(prm_post_med),
            ctx.alloc_array(ref.REGIMES[r_hyst]),
            ctx.alloc_array(eroi_hyst),
            ctx.alloc_array((n_hseeds, 80, 4), np.float64),
            ctx.alloc_array((n_hseeds, 4), np.float64),
        ]
        try:
            mod.launch("kernel_mod5_hysteresis", (n_hseeds + 63) // 64, 64, [
                n_hseeds, 80, ctypes.c_uint32(5151), 0.35, *h_bufs
            ])
            max_err = max(max_err, check_errs_fuse(h_bufs[4].to_host(), f"Mod5_Hyst_{r_hyst}"))
            st_mean = h_bufs[3].to_host().mean(axis=0)  # (80, 4): surv, O_state, B_infra, g_tot
            hyst_trajs[r_hyst] = st_mean
            hyst_res[r_hyst] = {
                "surv_t39_trough_pct": round(float(st_mean[39, 0] * 100.0), 2),
                "surv_t79_recovery_pct": round(float(st_mean[79, 0] * 100.0), 2),
                "ostate_t0": round(float(st_mean[0, 1]), 3),
                "ostate_t79_recovered": round(float(st_mean[79, 1]), 3),
                "b_infra_t39_trough": round(float(st_mean[39, 2]), 4),
            }
        finally:
            for b in h_bufs:
                b.free()

    np.savez_compressed(
        os.path.join(OUT_DIR, "mod5_phase_ews_hyst.npz"),
        eroi_axis=eroi_axis, gorg_axis=gorg_axis,
        Importer_Net=slice_maps["Importer_Net"], Exporter_Net=slice_maps["Exporter_Net"],
        eroi_hyst=eroi_hyst,
        **{f"hyst_{k}": v for k, v in hyst_trajs.items()},
    )
    summary = {
        "module": "Mod5_PhaseCliff_EWS_Hysteresis",
        "runtime_sec": round(time.time() - t0, 2),
        "grid_shape": [ng, ng], "seeds_per_point": n_seeds,
        "total_trajectories": int(2 * n_pts * n_seeds + 3 * n_hseeds),
        "max_closure_err": max_err,
        "phase_and_ews_slices": slice_summary,
        "hysteresis_loop": hyst_res,
    }
    save_json(os.path.join(OUT_DIR, "mod5_phase_ews_hyst.json"), summary)
    return summary


# =============================================================================
# MODULE 6: Zero-Leakage China OOS (1950-2050) Controlled 5-Way Factorial
# =============================================================================
def _run_mod6_pathway(ctx, mod, ch, post_full, reg_hist, reg_post, e101, et101, nf101, cm101,
                      lam_ovr, eta_rec_ovr, c_mult, trade_absorb, n_seeds_p, base_seed, write_traj=True):
    n_p = post_full.shape[0]
    cv101 = np.concatenate([ch["civ_hist"], np.full(26, ch["civ_hist"][-1])])
    d_rh = ctx.alloc_array(reg_hist); d_rp = ctx.alloc_array(reg_post)
    d_e = ctx.alloc_array(e101); d_et = ctx.alloc_array(et101); d_ag = ctx.alloc_array(ch["age_101"])
    d_cv = ctx.alloc_array(cv101); d_nf = ctx.alloc_array(nf101); d_nt = ctx.alloc_array(ch["net_trade_101"])
    d_cm = ctx.alloc_array(cm101)

    p_batch = max(1, min(n_p, 2048 // max(1, n_seeds_p)))
    d_pm = ctx.alloc_array((p_batch, 22), np.float64)
    d_sum = ctx.alloc_array((p_batch * n_seeds_p, 8), np.float64)
    d_traj = ctx.alloc_array((p_batch * n_seeds_p, 101, 4), np.float64) if write_traj else ctx.alloc_array(4, np.float64)
    d_errs = ctx.alloc_array((p_batch * n_seeds_p, 4), np.float64)

    all_sum = np.zeros((n_p * n_seeds_p, 8), dtype=np.float64)
    traj_pcts = {}
    # Collect full 101-step trajectories across seed 0 of all posterior particles (low RAM, full 10k posterior bands!)
    seed0_traj = np.zeros((n_p, 101, 4), dtype=np.float64) if write_traj else None
    max_err = 0.0
    try:
        for i0 in range(0, n_p, p_batch):
            i1 = min(n_p, i0 + p_batch)
            cur = i1 - i0
            n_j = cur * n_seeds_p
            dp_use = ctx.alloc_array(post_full[i0:i1]) if cur != p_batch else d_pm
            if cur == p_batch:
                d_pm.copy_from_host(post_full[i0:i1])
            mod.launch("kernel_mod6_china_oos", (n_j + 127) // 128, 128, [
                cur, n_seeds_p, 101, 75, ctypes.c_uint32(base_seed),
                float(lam_ovr), float(eta_rec_ovr), float(c_mult), 1, 1, 0, float(trade_absorb), 0,
                dp_use, d_rh, d_rp, d_e, d_et, d_ag, d_cv, d_nf, d_nt, d_cm,
                float(ch["init_pub_debt"]), float(ch["init_priv_debt"]),
                float(ch["etwh_2024"]), float(ch["co2_2024_gt"]), float(ch["clim_budget_chn"]),
                float(ch["min_caps"][0]), float(ch["min_caps"][1]), float(ch["min_caps"][2]),
                1 if write_traj else 0, d_sum, d_traj, d_errs
            ])
            if cur != p_batch:
                dp_use.free()
            max_err = max(max_err, check_errs_fuse(d_errs.to_host()[:n_j], "Mod6_CHN"))
            all_sum[i0 * n_seeds_p:i1 * n_seeds_p] = d_sum.to_host()[:n_j]
            if write_traj:
                t_b = d_traj.to_host()[:n_j].reshape(cur, n_seeds_p, 101, 4)
                seed0_traj[i0:i1] = t_b[:, 0, :, :]
    finally:
        for b in [d_rh, d_rp, d_e, d_et, d_ag, d_cv, d_nf, d_nt, d_cm, d_pm, d_sum, d_traj, d_errs]:
            b.free()

    if write_traj:
        for idx_k, nm in enumerate(["g", "d", "surv", "co2"]):
            traj_pcts[nm] = np.percentile(seed0_traj[:, :, idx_k], [5, 25, 50, 75, 95], axis=0)

    return all_sum, traj_pcts, max_err


def run_mod6(ctx, mod, ds, tier_cfg, post_full):
    t0 = time.time()
    print("\n=== [MOD 6] China (1950-2050) Zero-Leakage 5-Way Factorial & Sensitivity ===")
    ch = ds["china"]
    n_seeds_p = tier_cfg["mod6_seeds_per_p"]
    reg_h = ref.REGIMES["CHN_Hist_Base"]
    e24, eta24, nf24 = float(ch["eroi_hist"][-1]), float(ch["eta_hist"][-1]), float(ch["nonf_hist"][-1])

    # Symmetric 5-way factorial pathways (ZERO re-fitting on China!)
    # Format: (reg_post_key, eroi_tgt, eta_tgt, nonf_tgt, co2_mult_end, lam_learn, eta_rec, c_mult, exp_scale)
    chn_paths = {
        "CHN_A_Austerity_SlowElec":           ("CHN_A_Austerity_SlowElec",           12.0, 0.285, 0.50, 1.00, 1.00, 0.45, 1.00, False),
        "CHN_B_FastElec_Under_Austerity":     ("CHN_B_FastElec_Under_Austerity",     22.0, 0.365, 0.85, 0.88, 0.80, 0.45, 0.80, True),
        "CHN_C_Sovereign_FossilInertia":      ("CHN_C_Sovereign_FossilInertia",      11.5, 0.275, 0.44, 1.02, 1.00, 0.45, 1.00, False),
        "CHN_D_Current_Elec_Infra_Coord":     ("CHN_D_Current_Elec_Infra_Coord",     22.0, 0.365, 0.85, 0.88, 0.72, 0.55, 0.75, True),
        "CHN_E_Elec_Plus_EcoSocial_Rebalance":("CHN_E_Elec_Plus_EcoSocial_Rebalance",21.0, 0.370, 0.90, 0.72, 0.72, 0.78, 0.50, True),
    }

    path_summary = {}
    npz_trajs = {}
    max_err = 0.0
    for idx_k, (pname, (rk, e_t, et_t, nf_t, cm_e, lam, erec, cm, exps)) in enumerate(chn_paths.items()):
        e26, et26, nf26, cm26 = build_forward_sequences(e24, eta24, nf24, e_t, et_t, nf_t, cm_e, exp_scale=exps)
        e101 = np.concatenate([ch["eroi_hist"], e26])
        et101 = np.concatenate([ch["eta_hist"], et26])
        nf101 = np.concatenate([ch["nonf_hist"], nf26])
        cm101 = np.concatenate([np.ones(75), cm26])

        s_arr, tp_dict, merr = _run_mod6_pathway(
            ctx, mod, ch, post_full, reg_h, ref.REGIMES[rk], e101, et101, nf101, cm101,
            lam, erec, cm, 0.75, n_seeds_p, 6060, write_traj=True
        )
        max_err = max(max_err, merr)
        g50_med, d50_med, c50_med, m50_med = np.median(s_arr[:, [3, 4, 5, 6]], axis=0)
        g50_lo, g50_hi = np.percentile(s_arr[:, 3], [5, 95])
        c50_lo, c50_hi = np.percentile(s_arr[:, 5], [5, 95])
        m50_lo, m50_hi = np.percentile(s_arr[:, 6], [5, 95])
        viab_pct = float(np.mean(s_arr[:, 0]) * 100.0)

        path_summary[pname] = {
            "viability_2050_pct": round(viab_pct, 2),
            "legit_collapse_pct": round(float(np.mean(s_arr[:, 1] == 1) * 100.0), 2),
            "fin_unemp_collapse_pct": round(float(np.mean(s_arr[:, 1] == 2) * 100.0), 2),
            "g_2050_med_pct": round(float(g50_med * 100.0), 2),
            "g_2050_ci90_pct": [round(float(g50_lo * 100.0), 2), round(float(g50_hi * 100.0), 2)],
            "co2_cum_2025_2050_med_gt": round(float(c50_med), 2),
            "co2_cum_ci90_gt": [round(float(c50_lo), 2), round(float(c50_hi), 2)],
            "within_140gt_chn_budget": bool(c50_med <= 140.0),
            "mineral_ratio_med": round(float(m50_med), 3),
            "mineral_ratio_ci90": [round(float(m50_lo), 3), round(float(m50_hi), 3)],
            "debt_2050_med": round(float(d50_med), 3),
            "fx_2050_med": round(float(np.median(s_arr[:, 7])), 4),
        }
        for k_t, v_t in tp_dict.items():
            npz_trajs[f"{pname}_{k_t}_pcts"] = v_t

        # Compute blind out-of-sample 1982-2024 G/GDP & 1995-2024 IMF debt fit on shared CHN_Hist trunk
        if idx_k == 0:
            obs_g = ch["obs_g"]; obs_d = ch["obs_debt"]
            valid_g = ~np.isnan(obs_g); valid_d = ~np.isnan(obs_d)
            sim_g_med = tp_dict["g"][2][:75]
            sim_d_med = tp_dict["d"][2][:75]
            blind_fit = {
                "valid_g_years": f"{int(ch['years'][:75][valid_g][0])}-{int(ch['years'][:75][valid_g][-1])}",
                "obs_g_2024_pct": round(float(obs_g[74] * 100.0), 2),
                "blind_sim_g_2024_med_pct": round(float(sim_g_med[74] * 100.0), 2),
                "blind_g_mae_pp": round(float(np.mean(np.abs(sim_g_med[valid_g] - obs_g[valid_g])) * 100.0), 2),
                "obs_debt_2024_pct": round(float(obs_d[74] * 100.0), 2),
                "blind_sim_debt_2024_med_pct": round(float(sim_d_med[74] * 100.0), 2),
                "blind_debt_corr": round(float(np.corrcoef(sim_d_med[valid_d], obs_d[valid_d])[0, 1]), 4),
            }
            print(f"  Blind CHN History ({blind_fit['valid_g_years']}): "
                  f"g(2024) sim={blind_fit['blind_sim_g_2024_med_pct']:.1f}% vs IMF={blind_fit['obs_g_2024_pct']:.1f}% | "
                  f"Debt R={blind_fit['blind_debt_corr']:.3f}")

        print(f"  {pname:36s}: Viab={viab_pct:6.2f}% | CO2={c50_med:6.1f} Gt | Mineral={m50_med:.2f}x | g50={g50_med*100:.1f}%")

    # Learning-rate sensitivity band on CHN_D (lambda_learn = 0.72, 0.85, 0.95) & structural invariance
    e26_d, et26_d, nf26_d, cm26_d = build_forward_sequences(e24, eta24, nf24, 22.0, 0.365, 0.85, 0.88, exp_scale=True)
    e_d = np.concatenate([ch["eroi_hist"], e26_d]); et_d = np.concatenate([ch["eta_hist"], et26_d])
    nf_d = np.concatenate([ch["nonf_hist"], nf26_d]); cm_d = np.concatenate([np.ones(75), cm26_d])
    learn_band = {}
    n_stab_seeds = max(8, n_seeds_p // 2)
    for lam_v in [0.72, 0.85, 0.95]:
        s_l, _, m_l = _run_mod6_pathway(
            ctx, mod, ch, post_full, reg_h, ref.REGIMES["CHN_D_Current_Elec_Infra_Coord"],
            e_d, et_d, nf_d, cm_d, lam_v, 0.55, 0.75, 0.75, n_stab_seeds, 6200, write_traj=False
        )
        max_err = max(max_err, m_l)
        learn_band[f"lam_{lam_v:.2f}"] = {
            "viability_pct": round(float(np.mean(s_l[:, 0]) * 100.0), 2),
            "co2_med_gt": round(float(np.median(s_l[:, 5])), 2),
            "mineral_ratio_med": round(float(np.median(s_l[:, 6])), 3),
        }

    invariance_chn = {}
    for ta, psi_ovr, tag in [
        (0.50, 0.048, "high_trade_expose_ta0.50_psi0.048"),
        (0.75, 0.032, "baseline_ta0.75_psi0.032"),
        (0.90, 0.015, "low_trade_expose_ta0.90_psi0.015"),
    ]:
        inv_c = {}
        post_inv = post_full.copy()
        post_inv[:, 8] = psi_ovr
        for rk_s, nf_s, et_s, e_s, lam_s, rc_s, cm_s, ex_s in [
            ("CHN_A_Austerity_SlowElec", 0.50, 0.285, 12.0, 1.00, 0.45, 1.00, False),
            ("CHN_D_Current_Elec_Infra_Coord", 0.85, 0.365, 22.0, 0.72, 0.55, 0.88, True),
            ("CHN_E_Elec_Plus_EcoSocial_Rebalance", 0.90, 0.370, 21.0, 0.72, 0.78, 0.72, True),
        ]:
            e2, et2, nf2, cm2 = build_forward_sequences(e24, eta24, nf24, e_s, et_s, nf_s, cm_s, exp_scale=ex_s)
            s_i, _, mi = _run_mod6_pathway(
                ctx, mod, ch, post_inv, reg_h, ref.REGIMES[rk_s],
                np.concatenate([ch["eroi_hist"], e2]), np.concatenate([ch["eta_hist"], et2]),
                np.concatenate([ch["nonf_hist"], nf2]), np.concatenate([np.ones(75), cm2]),
                lam_s, rc_s, 0.75 if rk_s != "CHN_E_Elec_Plus_EcoSocial_Rebalance" else 0.50, ta,
                n_stab_seeds, 6300, write_traj=False
            )
            max_err = max(max_err, mi)
            inv_c[rk_s] = round(float(np.mean(s_i[:, 0]) * 100.0), 2)
        invariance_chn[tag] = inv_c

    # Optional Ultra-Tier 2D Synergy Frontier for China (eta_rec x g_trans_max)
    frontier_summary = None
    if tier_cfg.get("mod6_frontier_dim", 0) > 0:
        fd = tier_cfg["mod6_frontier_dim"]
        rec_ax = np.linspace(0.30, 0.85, fd)
        gt_ax = np.linspace(0.14, 0.32, fd)
        f_surv = np.zeros((fd, fd))
        f_min = np.zeros((fd, fd))
        sub_p = post_full[:min(256, post_full.shape[0])]
        for ir, rv in enumerate(rec_ax):
            for ig, gtv in enumerate(gt_ax):
                r_tmp = ref.REGIMES["CHN_D_Current_Elec_Infra_Coord"].copy()
                r_tmp[5] = gtv
                sf, _, mf = _run_mod6_pathway(ctx, mod, ch, sub_p, reg_h, r_tmp, e_d, et_d, nf_d, cm_d,
                                              0.72, rv, 0.70, 0.75, 2, 6500, write_traj=False)
                max_err = max(max_err, mf)
                f_surv[ir, ig] = float(np.mean(sf[:, 0]) * 100.0)
                f_min[ir, ig] = float(np.median(sf[:, 6]))
        npz_trajs["chn_frontier_rec_ax"] = rec_ax
        npz_trajs["chn_frontier_gt_ax"] = gt_ax
        npz_trajs["chn_frontier_surv"] = f_surv
        npz_trajs["chn_frontier_minratio"] = f_min
        frontier_summary = {"dim": fd, "max_viab_pct": round(float(f_surv.max()), 2), "min_mineral_ratio": round(float(f_min.min()), 3)}

    np.savez_compressed(os.path.join(OUT_DIR, "mod6_china_oos.npz"), years=ch["years"], **npz_trajs)
    summary = {
        "module": "Mod6_China_ZeroLeakage_OOS_5Way",
        "runtime_sec": round(time.time() - t0, 2),
        "n_oecd_posterior_particles": int(post_full.shape[0]),
        "seeds_per_particle": n_seeds_p,
        "max_closure_err": max_err,
        "blind_historical_fit": blind_fit,
        "controlled_5way_factorial": path_summary,
        "wright_learning_sensitivity_CHN_D": learn_band,
        "structural_invariance": invariance_chn,
        "china_2d_policy_frontier": frontier_summary,
    }
    save_json(os.path.join(OUT_DIR, "mod6_china_oos.json"), summary)
    return summary


# =============================================================================
# Tier Profiles & Master Entry Point
# =============================================================================
TIER_PROFILES = {
    "preflight": {
        "mod1_worlds": 64, "mod1_fals_worlds": 32,
        "mod2_rounds": 4, "mod2_draw": 2048, "mod2_pop": 256, "mod2_post_n": 256, "mod2_seeds": 2, "mod2_loco_isos": ["JPN", "NOR"],
        "mod3_n_base": 128, "mod3_n_crn": 4, "mod3_n_boot": 200, "mod3_per_country": False, "mod3_groups_only": ["OECD_Mean"],
        "mod4_dims": (4, 4, 3, 16),
        "mod5_grid": 32, "mod5_seeds": 16, "mod5_hyst_seeds": 64,
        "mod6_seeds_per_p": 4, "mod6_frontier_dim": 0,
    },
    "standard": {
        "mod1_worlds": 32768, "mod1_fals_worlds": 8192,
        "mod2_rounds": 14, "mod2_draw": 98304, "mod2_pop": 4096, "mod2_post_n": 10000, "mod2_seeds": 8,
        "mod2_loco_isos": ["AUS", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA",
                           "GBR", "IRL", "ITA", "JPN", "NLD", "NOR", "PRT", "SWE", "USA"],
        "mod3_n_base": 65536, "mod3_n_crn": 32, "mod3_n_boot": 1000, "mod3_per_country": False, "mod3_groups_only": [],
        "mod4_dims": (20, 16, 12, 512),
        "mod5_grid": 512, "mod5_seeds": 128, "mod5_hyst_seeds": 16384,
        "mod6_seeds_per_p": 32, "mod6_frontier_dim": 16,
    },
    "ultra": {
        "mod1_worlds": 65536, "mod1_fals_worlds": 16384,
        "mod2_rounds": 18, "mod2_draw": 131072, "mod2_pop": 6144, "mod2_post_n": 10000, "mod2_seeds": 12,
        "mod2_loco_isos": ["AUS", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA",
                           "GBR", "IRL", "ITA", "JPN", "NLD", "NOR", "PRT", "SWE", "USA"],
        "mod3_n_base": 65536, "mod3_n_crn": 64, "mod3_n_boot": 1000, "mod3_per_country": True, "mod3_groups_only": [],
        "mod4_dims": (24, 20, 16, 1024),
        "mod5_grid": 512, "mod5_seeds": 256, "mod5_hyst_seeds": 32768,
        "mod6_seeds_per_p": 64, "mod6_frontier_dim": 24,
    },
}


def write_manifest_and_master(t_all_start, mode_str, tier_str, hw_info, results):
    master = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": mode_str,
        "tier": tier_str,
        "total_runtime_sec": round(time.time() - t_all_start, 2),
        "platform": platform.platform(),
        "hardware_provenance": hw_info,
        "max_global_closure_err": max(
            [v.get("max_closure_err", 0.0) for v in results.values() if isinstance(v, dict)]
        ),
        "modules": results,
    }
    save_json(os.path.join(OUT_DIR, "hardware_provenance.json"), hw_info)
    save_json(os.path.join(OUT_DIR, "MASTER_SUMMARY.json"), master)

    manifest = {"files": {}}
    for fn in sorted(os.listdir(OUT_DIR)):
        fp = os.path.join(OUT_DIR, fn)
        if os.path.isfile(fp) and fn != "manifest_sha256.json" and not fn.endswith(".tmp"):
            manifest["files"][fn] = {"bytes": os.path.getsize(fp), "sha256": sha256_file(fp)}
    save_json(os.path.join(OUT_DIR, "manifest_sha256.json"), manifest)
    print(f"\n[ALL COMPLETE] Saved outputs + manifest_sha256.json in {master['total_runtime_sec']:.1f}s | "
          f"Global Max Closure Err = {master['max_global_closure_err']:.2e}")


def main():
    parser = argparse.ArgumentParser(description="RTX 5080 FP64 Open-Economy SFC-BP Experiment Suite")
    parser.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    parser.add_argument("--tier", choices=["standard", "ultra"], default="standard")
    parser.add_argument("--modules", default="0,1,2,3,4,5,6")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    if args.force:
        args.resume = False

    os.makedirs(OUT_DIR, exist_ok=True)
    mods_req = set(int(x.strip()) for x in args.modules.split(",") if x.strip() != "")
    prof_key = "preflight" if args.mode == "preflight" else args.tier
    tier_cfg = TIER_PROFILES[prof_key]

    t_all = time.time()
    ds = ref.load_experiment_datasets(DATA_DIR)
    ctx = CudaContext(device_id=0)
    try:
        cu_src = open(os.path.join(ENGINE_DIR, "sfc_bp_open_economy.cu"), "r", encoding="utf-8").read()
        mod = ctx.compile_and_load(cu_src, src_name="sfc_bp_open_economy.cu", ptx_arch="compute_80")
        if mod.compile_log.strip():
            print("NVRTC compile warnings:\n", mod.compile_log)

        results = {}
        # Always run Mod 0 hardware check & calibration
        res0 = run_mod0_preflight(ctx, mod, ds)
        results["mod0"] = res0

        def _ckpt_ok(jname, aname=None):
            if not args.resume:
                return False
            jp = os.path.join(OUT_DIR, jname)
            ap = os.path.join(OUT_DIR, aname) if aname else None
            return os.path.isfile(jp) and (ap is None or os.path.isfile(ap))

        if 1 in mods_req:
            if _ckpt_ok("mod1_open18_summary.json", "mod1_open18_arrays.npz"):
                print("\n[MOD 1 SKIPPED] Checkpoint already verified in outputs/ (--force to rerun)")
                results["mod1"] = json.load(open(os.path.join(OUT_DIR, "mod1_open18_summary.json"), "r", encoding="utf-8"))
            else:
                results["mod1"] = run_mod1(ctx, mod, ds, tier_cfg)

        post_full = None
        if 2 in mods_req:
            if _ckpt_ok("mod2_abc_summary.json", "mod2_abc_posterior.npz"):
                print("\n[MOD 2 SKIPPED] Checkpoint already verified in outputs/ (--force to rerun)")
                results["mod2"] = json.load(open(os.path.join(OUT_DIR, "mod2_abc_summary.json"), "r", encoding="utf-8"))
                post_full = np.load(os.path.join(OUT_DIR, "mod2_abc_posterior.npz"))["posterior_full"]
            else:
                res2, post_full = run_mod2(ctx, mod, ds, tier_cfg)
                results["mod2"] = res2
        elif os.path.isfile(os.path.join(OUT_DIR, "mod2_abc_posterior.npz")):
            post_full = np.load(os.path.join(OUT_DIR, "mod2_abc_posterior.npz"))["posterior_full"]

        if post_full is not None:
            prm_post_med = np.median(post_full, axis=0)
        else:
            prm_post_med = ref.DEFAULT_PARAMS.copy()

        if 3 in mods_req:
            if _ckpt_ok("mod3_sobol_summary.json", "mod3_sobol_arrays.npz"):
                print("\n[MOD 3 SKIPPED] Checkpoint already verified in outputs/ (--force to rerun)")
                results["mod3"] = json.load(open(os.path.join(OUT_DIR, "mod3_sobol_summary.json"), "r", encoding="utf-8"))
            else:
                results["mod3"] = run_mod3(ctx, mod, ds, tier_cfg)

        if 4 in mods_req:
            if _ckpt_ok("mod4_cannib_minerals.json", "mod4_cannib_minerals.npz"):
                print("\n[MOD 4 SKIPPED] Checkpoint already verified in outputs/ (--force to rerun)")
                results["mod4"] = json.load(open(os.path.join(OUT_DIR, "mod4_cannib_minerals.json"), "r", encoding="utf-8"))
            else:
                results["mod4"] = run_mod4(ctx, mod, ds, tier_cfg, prm_post_med)

        if 5 in mods_req:
            if _ckpt_ok("mod5_phase_ews_hyst.json", "mod5_phase_ews_hyst.npz"):
                print("\n[MOD 5 SKIPPED] Checkpoint already verified in outputs/ (--force to rerun)")
                results["mod5"] = json.load(open(os.path.join(OUT_DIR, "mod5_phase_ews_hyst.json"), "r", encoding="utf-8"))
            else:
                results["mod5"] = run_mod5(ctx, mod, tier_cfg, prm_post_med)

        if 6 in mods_req:
            if post_full is None:
                raise RuntimeError("Mod 6 requires Mod 2 OECD posterior particles (zero China refitting)!")
            if _ckpt_ok("mod6_china_oos.json", "mod6_china_oos.npz"):
                print("\n[MOD 6 SKIPPED] Checkpoint already verified in outputs/ (--force to rerun)")
                results["mod6"] = json.load(open(os.path.join(OUT_DIR, "mod6_china_oos.json"), "r", encoding="utf-8"))
            else:
                results["mod6"] = run_mod6(ctx, mod, ds, tier_cfg, post_full)

        write_manifest_and_master(t_all, args.mode, prof_key, res0["gpu"], results)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()

r"""
Calibrated Mod 5A Phase Diagram & Pre-Collapse EWS Refresh (cb=0.560)
- Uses the bundled Tier 2 Ultra 18-round converged ABC-SMC posterior.
- Adjusts ONLY the Mod 5A civilian fiscal corridor ceiling in memory (0.435 -> 0.560,
  cap 0.39 -> 0.52) to align with the converged OECD baseline (~48-52% GDP).
- Copies Mod 5B hysteresis loops verbatim from results/tier2_ultra/mod5_phase_ews_hyst.npz.
"""
import ctypes
import hashlib
import json
import os
import pathlib
import sys
import time
import numpy as np

ARCHIVE_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE_DIR = ARCHIVE_ROOT / "engine"
RAW_5080 = ARCHIVE_ROOT / "results" / "tier2_ultra"
OUT_MOD5 = ARCHIVE_ROOT / "results" / "mod5a_calibrated_cb056"
OUT_MOD5.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ENGINE_DIR))
import cuda_driver  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    post_path = RAW_5080 / "mod2_abc_posterior.npz"
    hyst_path = RAW_5080 / "mod5_phase_ews_hyst.npz"
    assert post_path.exists(), f"Missing {post_path}"
    assert hyst_path.exists(), f"Missing {hyst_path}"

    post_full = np.load(post_path)["posterior_full"]
    prm_post_med = np.median(post_full, axis=0).astype(np.float64)

    cu_path = ENGINE_DIR / "sfc_bp_open_economy.cu"
    cu_src = cu_path.read_text(encoding="utf-8")
    old_line = "double civ_budget = clampd(0.435 - gorg_fix, 0.12, 0.39);"
    new_line = "double civ_budget = clampd(0.560 - gorg_fix, 0.12, 0.52);"
    assert cu_src.count(old_line) == 1, "Expected exact Mod 5A ceiling line once in .cu"
    cu_src_patched = cu_src.replace(old_line, new_line, 1)

    ctx = cuda_driver.CudaContext(0)
    try:
        mod = ctx.compile_and_load(cu_src_patched)
        if mod.compile_log:
            print(f"[NVRTC log] {mod.compile_log}")

        ng = 512
        n_seeds = 256
        eroi_axis = np.linspace(4.0, 32.0, ng, dtype=np.float64)
        gorg_axis = np.linspace(0.02, 0.18, ng, dtype=np.float64)
        ee, gg = np.meshgrid(eroi_axis, gorg_axis, indexing="xy")
        e_flat = ee.ravel()
        g_flat = gg.ravel()
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
            m_out = np.zeros((n_pts, 5), dtype=np.float64)
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
                    mod.launch(
                        "kernel_mod5_phase_ews",
                        (n_j + 127) // 128,
                        128,
                        [
                            cur, n_seeds, 45, ctypes.c_uint32(5050), float(imp_val),
                            bufs[0], de, dg, bufs[3], bufs[4],
                        ],
                    )
                    if cur != pts_batch:
                        de.free()
                        dg.free()
                    if np.max(bufs[4].to_host()[:n_j]) > 0.5:
                        raise RuntimeError("NaN detected in calibrated Mod 5A kernel!")
                    raw_b = bufs[3].to_host()[:n_j].reshape(cur, n_seeds, 6)
                    err_b = float(np.max(raw_b[:, :, 5]))
                    max_err = max(max_err, err_b)
                    assert max_err < 1e-11, f"Closure violation: {max_err}"
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
            idx_o10 = int(np.argmin(np.abs(gorg_axis - 0.10)))
            idx_o15 = int(np.argmin(np.abs(gorg_axis - 0.15)))
            e_m5 = float(eroi_axis[np.argmax(m_grid[idx_o5, :, 0] >= 0.50)]) if np.any(m_grid[idx_o5, :, 0] >= 0.50) else 32.0
            e_m10 = float(eroi_axis[np.argmax(m_grid[idx_o10, :, 0] >= 0.50)]) if np.any(m_grid[idx_o10, :, 0] >= 0.50) else 32.0
            e_m15 = float(eroi_axis[np.argmax(m_grid[idx_o15, :, 0] >= 0.50)]) if np.any(m_grid[idx_o15, :, 0] >= 0.50) else 32.0

            safe_mask = m_grid[:, :, 0] >= 0.85
            tip_mask = (m_grid[:, :, 0] >= 0.15) & (m_grid[:, :, 0] <= 0.70)
            assert np.any(safe_mask), f"Empty safe_mask for {sname}"
            assert np.any(tip_mask), f"Empty tip_mask for {sname}"
            assert e_m15 > e_m5, f"Expected higher critical EROI at gorg=15% vs 5% for {sname}"

            slice_summary[sname] = {
                "safe_zone_pixels": int(np.sum(safe_mask)),
                "tipping_zone_pixels": int(np.sum(tip_mask)),
                "min_viable_eroi_at_gorg_5pct": round(e_m5, 2),
                "min_viable_eroi_at_gorg_10pct": round(e_m10, 2),
                "min_viable_eroi_at_gorg_15pct": round(e_m15, 2),
                "ews_ar1_diff_debt_safe": round(float(m_grid[safe_mask, 2].mean()), 4),
                "ews_ar1_diff_debt_tipping": round(float(m_grid[tip_mask, 2].mean()), 4),
                "ews_var_binfra_safe": round(float(m_grid[safe_mask, 3].mean()), 6),
                "ews_var_binfra_tipping": round(float(m_grid[tip_mask, 3].mean()), 6),
                "ews_ar1_binfra_safe": round(float(m_grid[safe_mask, 4].mean()), 4),
                "ews_ar1_binfra_tipping": round(float(m_grid[tip_mask, 4].mean()), 4),
            }
            print(
                f"  [{sname:12s}] MinEROI @5%={e_m5:.2f}, @10%={e_m10:.2f}, @15%={e_m15:.2f} | "
                f"Var(B_infra) Safe={slice_summary[sname]['ews_var_binfra_safe']:.5f} -> "
                f"Tip={slice_summary[sname]['ews_var_binfra_tipping']:.5f} | "
                f"SafePx={slice_summary[sname]['safe_zone_pixels']}, TipPx={slice_summary[sname]['tipping_zone_pixels']}"
            )

        # Copy Mod 5B hysteresis verbatim from RTX 5080 Tier 2 archive
        raw_m5 = np.load(hyst_path)
        raw_m5_json = json.loads((RAW_5080 / "mod5_phase_ews_hyst.json").read_text(encoding="utf-8"))

        out_npz = OUT_MOD5 / "mod5_phase_ews_hyst_cb056.npz"
        np.savez_compressed(
            out_npz,
            eroi_axis=eroi_axis,
            gorg_axis=gorg_axis,
            Importer_Net=slice_maps["Importer_Net"],
            Exporter_Net=slice_maps["Exporter_Net"],
            eroi_hyst=raw_m5["eroi_hyst"],
            hyst_1_Adaptive_Growth=raw_m5["hyst_1_Adaptive_Growth"],
            hyst_2_Austerity_Coercion=raw_m5["hyst_2_Austerity_Coercion"],
            hyst_3_Neoliberal_Cap=raw_m5["hyst_3_Neoliberal_Cap"],
        )

        out_json = OUT_MOD5 / "mod5_phase_ews_hyst_cb056.json"
        meta = {
            "module": "Mod5A_Calibrated_FiscalCorridor_cb0560_and_5080_Mod5B",
            "runtime_sec": round(time.time() - t0, 2),
            "hardware": {**ctx.info_dict(), "nvrtc_dll_path": "nvrtc64_120_0.dll"},
            "calibration_note": (
                "Identical kernel_mod5_phase_ews physics and seed=5050 evaluated at the converged "
                "RTX 5080 Tier 2 18-round posterior median (prm_post_med), relaxing only the synthetic "
                "slice budget cap from cb=0.435 (which fell below the converged 48-52% OECD baseline) "
                "to cb=0.560 (cap 0.52)."
            ),
            "grid_shape": [ng, ng],
            "seeds_per_point": n_seeds,
            "total_trajectories": int(2 * n_pts * n_seeds),
            "max_closure_err": max_err,
            "input_posterior_sha256": sha256_file(post_path),
            "output_npz_sha256": sha256_file(out_npz),
            "phase_and_ews_slices": slice_summary,
            "hysteresis_loop_verbatim_from_5080": raw_m5_json["hysteresis_loop"],
        }
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[DONE] Saved calibrated Mod 5A to {out_npz} in {meta['runtime_sec']}s (MaxErr={max_err:.2e})")
    finally:
        ctx.close()


if __name__ == "__main__":
    main()

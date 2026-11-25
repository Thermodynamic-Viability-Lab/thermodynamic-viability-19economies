# Replication Code and Experimental Archive (Double-Anonymous Review Release v1.0)

This self-contained archive provides all simulation kernels, bit-exact CPU reference solvers, curated historical datasets, verified Tier 1 and Tier 2 RTX 5080 outputs, calibrated Mod 5A outputs (`c_b = 0.560`), and relative-path plotting scripts accompanying the manuscript:

> **Thermodynamic viability corridors and institutional selection across nineteen economies (1920-2050)**

## Directory Layout

- `engine/`: Core FP64 Open-Economy 5-Sector + Rest-of-World Stock-Flow-Fund-Biophysical simulation engine (`sfc_bp_open_economy.cu`), bit-exact Python CPU mirror (`cpu_reference.py`), NVRTC GPU driver (`cuda_driver.py`), statistical wrappers (`statistical_methods.py`), and panel loader (`prepare_input_data.py`).
- `data/`: Compiled 18-OECD historical panel (`1920-2024`), isolated China out-of-sample panel (`1950-2024`), UN DESA WPP 2024 old-age dependency projections (`1920-2050`), raw IMF WEO April 2025 / FRED extracts, and `data_provenance_sha256.json`.
- `results/tier2_ultra/`: Complete Tier 2 Ultra RTX 5080 outputs (`>5.5 billion` trajectories, `manifest_sha256.json`).
- `results/tier1_standard/`: Independent Tier 1 Standard RTX 5080 cross-check outputs (`manifest_sha256.json`).
- `results/mod5a_calibrated_cb056/`: Calibrated `512 x 512` Mod 5A phase & early-warning output (`c_b = 0.560`, `134.2M` trajectories, `manifest_sha256.json`).
- `analysis_scripts/`: Portable relative-path scripts to regenerate all 5 manuscript figures (`generate_nature_sust_figures.py`) or re-run the calibrated Mod 5A phase diagram on an NVIDIA GPU (`run_mod5a_calibrated_refresh.py`).
- `verify_archive_integrity.py`: Cryptographic SHA-256 audit, headline metric cross-check, and fast CPU conservation test (`< 15 s`, no GPU required).
- `verify_cpu_closure_fast.py`: Standalone FP64 CPU conservation verifier across all 18 OECD economies + China.
- `run_rtx5080_suite.py`: Full multi-tier CUDA/NVRTC GPU benchmark runner (Modules 0-6).

## Quick Verification (CPU Only, ~15 Seconds)

```bash
pip install -r requirements.txt
python verify_archive_integrity.py
```

`verify_archive_integrity.py` checks 100% of SHA-256 digests across `data/` and all three `results/` folders, prints headline statistics from `MASTER_SUMMARY.json` and `mod5_phase_ews_hyst_cb056.json`, and executes `verify_cpu_closure_fast.py` confirming exact double-precision conservation (`< 1e-12`).

## Regenerate All 5 Manuscript Figures (CPU Only, ~20 Seconds)

```bash
python analysis_scripts/generate_nature_sust_figures.py
```

Outputs are written to `reproduced_figures_600dpi/` at 600 DPI (PNG).

## Optional GPU Execution (NVIDIA Ampere / Ada / Blackwell, CUDA 12+)

```bash
python analysis_scripts/run_mod5a_calibrated_refresh.py
python run_rtx5080_suite.py --tier 1 --out-dir results/reproduced_tier1
```

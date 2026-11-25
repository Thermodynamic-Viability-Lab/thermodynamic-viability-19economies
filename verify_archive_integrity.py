"""
Standalone Reviewer Integrity & Conservation Verifier (Pure CPU, < 15 seconds).
Checks SHA-256 digests across empirical inputs and all bundled GPU result archives,
prints headline metrics, and executes verify_cpu_closure_fast.py.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(folder: pathlib.Path, manifest_name: str):
    mf_path = folder / manifest_name
    data = json.loads(mf_path.read_text(encoding="utf-8"))
    files_map = data["files"]
    for fname, meta in files_map.items():
        fp = folder / fname
        assert fp.exists(), f"Missing file in archive: {fp}"
        actual = sha256_file(fp)
        expected = meta["sha256"]
        assert actual == expected, f"SHA-256 mismatch for {fp}: {actual} != {expected}"
    return len(files_map)


def main():
    n_data = verify_manifest(ROOT / "data", "data_provenance_sha256.json")
    n_t2 = verify_manifest(ROOT / "results" / "tier2_ultra", "manifest_sha256.json")
    n_t1 = verify_manifest(ROOT / "results" / "tier1_standard", "manifest_sha256.json")
    n_m5 = verify_manifest(ROOT / "results" / "mod5a_calibrated_cb056", "manifest_sha256.json")
    print(f"[SHA-256 OK] data={n_data}, tier2_ultra={n_t2}, tier1_standard={n_t1}, mod5a_calibrated={n_m5}")

    ms = json.loads((ROOT / "results" / "tier2_ultra" / "MASTER_SUMMARY.json").read_text(encoding="utf-8"))
    m5 = json.loads((ROOT / "results" / "mod5a_calibrated_cb056" / "mod5_phase_ews_hyst_cb056.json").read_text(encoding="utf-8"))
    m1 = ms["modules"]["mod1"]["historical_regimes"]
    m2 = ms["modules"]["mod2"]
    m6 = ms["modules"]["mod6"]
    print(
        f"[HEADLINE STATS] GlobalMaxErr={ms['max_global_closure_err']:.2e} | "
        f"Mod1 Adaptive={m1['1_Adaptive_Growth']['oecd_viability_pct']:.2f}% | "
        f"Mod2 EmergentViab={m2['round_schedule'][-1]['emergent_viab_pct']:.2f}% | "
        f"Mod5A Imp@5%={m5['phase_and_ews_slices']['Importer_Net']['min_viable_eroi_at_gorg_5pct']:.2f} | "
        f"Mod6 CHN_D={m6['controlled_5way_factorial']['CHN_D_Current_Elec_Infra_Coord']['viability_2050_pct']:.2f}% | "
        f"CHN_E={m6['controlled_5way_factorial']['CHN_E_Elec_Plus_EcoSocial_Rebalance']['viability_2050_pct']:.2f}%"
    )

    res = subprocess.run([sys.executable, str(ROOT / "verify_cpu_closure_fast.py")], check=False)
    assert res.returncode == 0, f"verify_cpu_closure_fast.py failed with code {res.returncode}"
    print("[ARCHIVE VERIFICATION PASSED]")


if __name__ == "__main__":
    main()

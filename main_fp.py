from pathlib import Path

from fingerprint.config import load_config
from fingerprint.fingerprint_builder import build_fingerprint
from fingerprint.fingerprint_analysis import run_fingerprint_analysis
from fingerprint.synthetic_data import generate_synthetic_trades


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trades, dimensions, client_persona, client_aum, _persona_names = (
        generate_synthetic_trades(n_clients=200, n=5000, seed=0, n_outlier_clients=20)
    )

    # fp feeds PCA/UMAP/HDBSCAN below; fp_raw is the business-legible
    # companion (actual trade counts, dollar amounts, plain shares) -- see
    # `fingerprint.FingerprintResult` and `docs/fingerprint-features.md`.
    fp, fp_raw = build_fingerprint(trades, "client_id", dimensions, aum=client_aum,
                                   alpha=10.0, keep=("count", "gap"))

    config = load_config(Path(__file__).parent / "fingerprint" / "config.yaml")
    run_fingerprint_analysis(fp, fp_raw, client_persona=client_persona,
                             client_aum=client_aum, config=config)

    fp_no_aum, fp_no_aum_raw = build_fingerprint(trades, "client_id", dimensions,
                                                 alpha=10.0, keep=("count", "gap"))
    run_fingerprint_analysis(fp_no_aum, fp_no_aum_raw, config=config)

    # Sweep example (ad hoc use, e.g. from a notebook):
    #   from fingerprint.config import sweep_configs
    #   from fingerprint.fingerprint_analysis import run_sweep
    #   sweep_df = run_sweep(fp, sweep_configs(config, pca={"n_components": [5, 10, 15]}),
    #                        client_persona=client_persona, client_aum=client_aum)

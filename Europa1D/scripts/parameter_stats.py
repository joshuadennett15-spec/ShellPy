import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def main() -> None:
    data = np.load(os.path.join(RESULTS_DIR, "monte_carlo_results.npz"))
    thickness = data["thicknesses_km"]

    params = {}
    for key in data.keys():
        if key.startswith("param_"):
            params[key[6:]] = data[key]

    if not params:
        raise RuntimeError("No sampled parameters found in monte_carlo_results.npz")

    rows = []
    for name, values in params.items():
        if values.shape[0] != thickness.shape[0]:
            continue

        rho_s, p_s = spearmanr(values, thickness)
        rho_p, p_p = pearsonr(values, thickness)

        rows.append(
            {
                "parameter": name,
                "spearman_r": rho_s,
                "spearman_p": p_s,
                "pearson_r": rho_p,
                "pearson_p": p_p,
            }
        )

    df = pd.DataFrame(rows).sort_values(by="spearman_r", key=np.abs, ascending=False)
    df.to_csv(os.path.join(RESULTS_DIR, "parameter_stats.csv"), index=False)

    # Print top 8 by absolute Spearman correlation
    top = df.head(8).copy()
    top["spearman_r"] = top["spearman_r"].map(lambda x: f"{x:.3f}")
    top["pearson_r"] = top["pearson_r"].map(lambda x: f"{x:.3f}")
    print("Top parameters by |Spearman r|")
    print(top[["parameter", "spearman_r", "pearson_r"]].to_string(index=False))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from bso.core import (
    compute_rsi, realized_vol, rolling_zscores,
    rolling_hist_var, ewma_sigma,
    rolling_isolation_forest, smooth_anomaly, apply_overlay,
    evaluate_var
)

try:
    import yfinance as yf
except Exception:
    yf = None

def load_data_yf(start, end):
    if yf is None:
        raise SystemExit("yfinance not installed. pip install yfinance")
    pr = yf.download("^GSPC", start=start, end=end)["Adj Close"].rename("Price")
    vx = yf.download("^VIX", start=start, end=end)["Adj Close"].rename("VIX")
    df = pd.concat([pr, vx], axis=1).dropna()
    df["PutCall"] = np.nan  # optional
    return df

def load_data_csv(path):
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
    required = {"Price","VIX"}
    if not required.issubset(df.columns):
        raise SystemExit("CSV must contain at least: Date, Price, VIX. Optional: PutCall")
    return df

def main():
    ap = argparse.ArgumentParser(description="Run BSO overlay on real or CSV data")
    ap.add_argument("--use-yf", action="store_true", help="Use yfinance to pull ^GSPC and ^VIX")
    ap.add_argument("--csv", type=str, default="", help="Path to CSV with Date,Price,VIX,[PutCall]")
    ap.add_argument("--start", type=str, default="2016-01-01")
    ap.add_argument("--end", type=str, default="2025-09-16")
    ap.add_argument("--var", type=str, choices=["hist","ewma"], default="hist")
    ap.add_argument("--alpha", type=float, default=0.3)
    args = ap.parse_args()

    if args.use_yf:
        df = load_data_yf(args.start, args.end)
    else:
        if not args.csv:
            raise SystemExit("Provide --csv or use --use-yf")
        df = load_data_csv(args.csv)
        df = df.loc[(df.index >= args.start) & (df.index <= args.end)]

    # Features
    df["returns"] = df["Price"].pct_change()
    df["rsi14"] = compute_rsi(df["Price"], 14)
    df["rv_20"]  = realized_vol(df["returns"], 20, annualize=True)
    df["vix_slope"] = df["VIX"] - df["VIX"].shift(20)
    if "PutCall" not in df.columns:
        df["PutCall"] = 0.6 + (-df["returns"].clip(upper=0) * 10)  # proxy

    z = rolling_zscores(df[["rsi14","vix_slope","PutCall","rv_20"]])

    # Baseline VaR
    if args.var == "hist":
        var_base = rolling_hist_var(df["returns"], q=0.01, window=252)
    else:
        sigma = ewma_sigma(df["returns"], lam=0.94)
        var_base = 2.326 * sigma

    # Unsupervised anomaly
    an = rolling_isolation_forest(z, train_win=504, step=1, contamination=0.02, random_state=42)
    an_sm, flags = smooth_anomaly(an, threshold=0.6, cap=0.15, confirm=3)

    # Overlay
    var_adj = apply_overlay(var_base, an_sm, alpha=args.alpha)

    # Evaluate
    eval_base, vio_base = evaluate_var(df["returns"], var_base, alpha=0.01)
    eval_adj,  vio_adj  = evaluate_var(df["returns"], var_adj,  alpha=0.01)

    print("\n=== Coverage Results (99% VaR) ===")
    print("Baseline:", eval_base)
    print("Adjusted:", eval_adj)

    cap_vol_base = float(var_base.dropna().pct_change().std())
    cap_vol_adj  = float(var_adj.dropna().pct_change().std())
    uplift = float((var_adj.mean() / var_base.mean() - 1) * 100)

    print("\n=== Capital Metrics ===")
    print(f"Mean VaR base: {var_base.mean():.6f}, Mean VaR adj: {var_adj.mean():.6f}, Uplift: {uplift:.2f}%")
    print(f"Capital vol (std pct-change): base {cap_vol_base:.4f}, adj {cap_vol_adj:.4f}")

    # Plots
    plt.figure(figsize=(11,4))
    df["Price"].plot(title="Price")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(11,3))
    an_sm.plot()
    plt.scatter(an_sm.index[flags==1], an_sm[flags==1], marker='x')
    plt.title("Anomaly Score (smoothed) & confirmed flags")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(11,3))
    var_base.plot(label="VaR_base")
    var_adj.plot(label="VaR_adj")
    plt.title("VaR Base vs Adjusted (BSO overlay)"); plt.legend()
    plt.tight_layout(); plt.show()

    out = pd.DataFrame({
        "Price": df["Price"], "VIX": df["VIX"], "PutCall": df["PutCall"],
        "returns": df["returns"],
        "var_base": var_base, "var_adj": var_adj,
        "anomaly_sm": an_sm, "anomaly_flag": flags
    })
    out_path = Path("bso_results.csv")
    out.to_csv(out_path)
    print(f"\nSaved results to {out_path.resolve()}")

if __name__ == "__main__":
    main()


# BSO Synthetic Demo (scriptified notebook)
# Run: python notebooks/BSO_synthetic_demo.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from bso.core import compute_rsi, realized_vol, rolling_zscores, rolling_hist_var, rolling_isolation_forest, smooth_anomaly, apply_overlay

np.random.seed(42)
N = 1000
dates = pd.bdate_range(end=pd.Timestamp("2025-09-16"), periods=N)
mu, sigma = 0.0002, 0.01
returns = np.random.normal(mu, sigma, N)
for i in range(50, N, 200):
    returns[i:i+3] += np.random.normal(-0.05, 0.02, size=3)
price = 1000 * np.exp(np.cumsum(returns))
df = pd.DataFrame(index=dates)
df["Price"] = price
df["returns"] = pd.Series(returns, index=dates)
# Simulate VIX and Put/Call proxies
vix = 12 + np.random.normal(0,1.2,N)
vix += np.abs(np.where(df['returns'] < -0.02, np.random.uniform(5,12,size=N), 0))
df["VIX"] = np.clip(vix, 8, None)
df["PutCall"] = 0.6 + (-df["returns"].clip(upper=0)*10) + np.random.normal(0,0.05,N)

# Features
df["rsi14"] = compute_rsi(df["Price"], 14)
df["rv_20"]  = realized_vol(df["returns"], 20, annualize=True)
df["vix_slope"] = df["VIX"] - df["VIX"].shift(20)
z = rolling_zscores(df[["rsi14","vix_slope","PutCall","rv_20"]])

# Baseline VaR
var_base = rolling_hist_var(df["returns"], q=0.01, window=252)

# Anomaly
an = rolling_isolation_forest(z, train_win=504, step=1, contamination=0.02, random_state=42)
an_sm, flags = smooth_anomaly(an, threshold=0.6, cap=0.15, confirm=3)

# Overlay
var_adj = apply_overlay(var_base, an_sm, alpha=0.3)

# Plots
plt.figure(figsize=(11,4)); df["Price"].plot(title="Price"); plt.tight_layout(); plt.show()
plt.figure(figsize=(11,3)); an_sm.plot(); plt.scatter(an_sm.index[flags==1], an_sm[flags==1], marker='x'); plt.title("Anomaly Score & flags"); plt.tight_layout(); plt.show()
plt.figure(figsize=(11,3)); var_base.plot(label="VaR_base"); var_adj.plot(label="VaR_adj"); plt.legend(); plt.title("VaR Base vs Adjusted"); plt.tight_layout(); plt.show()

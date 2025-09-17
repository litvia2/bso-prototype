
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------- Feature Engineering ---------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def realized_vol(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    rv = returns.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv

def rolling_zscores(df: pd.DataFrame, window: int = 252, min_periods: int = 30) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        m = df[c].rolling(window, min_periods=min_periods).mean()
        s = df[c].rolling(window, min_periods=min_periods).std().replace(0, np.nan)
        out[c + "_z"] = ((df[c] - m) / s).fillna(0.0)
    return out

# --------- Baseline VaR ---------
def rolling_hist_var(returns: pd.Series, q: float = 0.01, window: int = 252) -> pd.Series:
    var = returns.rolling(window).quantile(q).abs()
    default = abs(returns.quantile(q))
    return var.fillna(default)

def ewma_sigma(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    var = np.zeros_like(returns.values, dtype=float)
    v = np.nanvar(returns.dropna().values) if returns.notna().any() else 0.0
    for i, r in enumerate(returns.fillna(0.0).values):
        v = lam * v + (1 - lam) * (r ** 2)
        var[i] = v
    return pd.Series(np.sqrt(var), index=returns.index)

# --------- Unsupervised Anomaly (rolling IsolationForest) ---------
def rolling_isolation_forest(z: pd.DataFrame, train_win: int = 504, step: int = 1, contamination: float = 0.02, random_state: int = 42) -> pd.Series:
    scaler = StandardScaler()
    X = z.values
    idx = z.index
    scores = pd.Series(index=idx, dtype=float)
    for t in range(train_win, len(X), step):
        X_train = X[t-train_win:t]
        X_test = X[t:t+step]
        Xtr = scaler.fit_transform(X_train)
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        iso.fit(Xtr)
        Xte = scaler.transform(X_test)
        sc = iso.score_samples(Xte)  # higher = more normal
        # convert to [0,1] anomaly using per-step min/max (simple, local scaling)
        sc_series = pd.Series(sc, index=idx[t:t+step])
        mn, mx = sc_series.min(), sc_series.max()
        if mx - mn == 0:
            an = pd.Series(0.0, index=sc_series.index)
        else:
            an = 1 - (sc_series - mn) / (mx - mn)
        scores.loc[an.index] = an
    return scores

def smooth_anomaly(score: pd.Series, threshold: float = 0.6, cap: float = 0.15, confirm: int = 3):
    score = score.copy().fillna(0.0)
    # cap daily changes
    sm = score.copy()
    for i in range(1, len(sm)):
        prev = sm.iloc[i-1]
        curr = sm.iloc[i]
        delta = curr - prev
        if delta > cap:
            sm.iloc[i] = prev + cap
        elif delta < -cap:
            sm.iloc[i] = prev - cap
    # confirmation flags
    f = (sm > threshold).astype(int)
    conf = f.copy()
    for i in range(confirm-1, len(f)):
        conf.iloc[i] = 1 if f.iloc[i-confirm+1:i+1].sum() == confirm else 0
    return sm, conf

# --------- Overlay ---------
def apply_overlay(var_base: pd.Series, anomaly_sm: pd.Series, alpha: float = 0.3) -> pd.Series:
    return var_base * (1 + alpha * anomaly_sm.fillna(0.0))

# --------- Evaluation ---------
from scipy.stats import chi2
def kupiec_pof(violations: pd.Series, alpha: float):
    n = len(violations)
    x = int(violations.sum())
    if n == 0:
        return np.nan, np.nan, x, n
    pi = x / n
    if pi in [0,1]:
        lr_pof = 0.0 if pi == alpha else 1e9
    else:
        lr_pof = -2 * ( (n - x)*np.log(1 - alpha) + x*np.log(alpha) - ( (n - x)*np.log(1 - pi) + x*np.log(pi) ) )
    pval = 1 - chi2.cdf(lr_pof, df=1)
    return lr_pof, pval, x, n

def christoffersen_independence(violations: pd.Series):
    v = violations.astype(int).values
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        if v[i-1] == 0 and v[i] == 0: n00 += 1
        elif v[i-1] == 0 and v[i] == 1: n01 += 1
        elif v[i-1] == 1 and v[i] == 0: n10 += 1
        elif v[i-1] == 1 and v[i] == 1: n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0
    def L(p, a, b):
        if p in [0,1]:
            if p == 0 and b>0: return -np.inf
            if p == 1 and a>0: return -np.inf
        return a*np.log(1 - p + 1e-12) + b*np.log(p + 1e-12)
    ll_ind = L(pi, n00 + n10, n01 + n11)
    ll_dep = L(pi0, n00, n01) + L(pi1, n10, n11)
    lr_ind = -2 * (ll_ind - ll_dep)
    pval = 1 - chi2.cdf(lr_ind, df=1)
    return lr_ind, pval, dict(n00=n00, n01=n01, n10=n10, n11=n11)

def evaluate_var(returns: pd.Series, var: pd.Series, alpha: float = 0.01):
    violations = (returns < -var).astype(int)
    kupiec = kupiec_pof(violations, alpha)
    christ = christoffersen_independence(violations)
    return {'kupiec': kupiec, 'christoffersen': christ, 'breaches': int(violations.sum()), 'n': int(len(violations))}, violations

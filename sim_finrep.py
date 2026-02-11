import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from pyod.models.vae import VAE

# ----------------------------
# 1) Simulated "FINREP-like" data
# ----------------------------
def simulate_finrep_like(
    n_samples=12000,
    templates=(("F01", 60), ("F02", 40), ("F18", 30), ("F20", 20)),
    n_latent=8,
    missing_rate=0.15,
    outlier_frac=0.04,
    seed=7,
):
    rng = np.random.default_rng(seed)

    # Feature layout (template blocks)
    template_names = []
    feature_template = []
    for tname, k in templates:
        template_names.append(tname)
        feature_template += [tname] * k
    feature_template = np.array(feature_template)
    n_features = len(feature_template)

    # Latent factors
    Z = rng.standard_normal((n_samples, n_latent))

    # Block loadings: each template depends more on a subset of factors
    W = np.zeros((n_latent, n_features))
    col = 0
    for t_idx, (tname, k) in enumerate(templates):
        # pick 3 dominant factors per template (overlapping allowed)
        dom = rng.choice(n_latent, size=min(3, n_latent), replace=False)
        base = rng.normal(0.0, 0.15, size=(n_latent, k))
        base[dom, :] += rng.normal(0.0, 0.8, size=(len(dom), k))
        W[:, col:col + k] = base
        col += k

    # Base signal + noise (allow negatives; FINREP has signed positions)
    X = Z @ W + rng.normal(0.0, 0.35, size=(n_samples, n_features))

    # Make a few "ratio-like" features by squashing with tanh (bounded-ish)
    ratio_idx = rng.choice(n_features, size=int(0.25 * n_features), replace=False)
    X[:, ratio_idx] = np.tanh(X[:, ratio_idx])

    # Inject structural missingness: template-specific + random
    # Template-specific: some banks "don't report" parts of certain templates
    miss = np.zeros_like(X, dtype=bool)
    for tname, _k in templates:
        cols = np.where(feature_template == tname)[0]
        # 10% of samples miss this template block (structural NA)
        rows = rng.choice(n_samples, size=int(0.10 * n_samples), replace=False)
        miss[np.ix_(rows, cols)] = True

    # Random missingness on top
    miss |= (rng.random(X.shape) < missing_rate)

    X_obs = X.copy()
    X_obs[miss] = np.nan

    # Create labeled outliers in the observed space before scaling
    y = np.zeros(n_samples, dtype=int)
    n_out = int(outlier_frac * n_samples)
    out_idx = rng.choice(n_samples, size=n_out, replace=False)
    y[out_idx] = 1

    # Outlier types
    # (a) Shift a few features
    for i in out_idx[: n_out // 4]:
        cols = rng.choice(n_features, size=rng.integers(3, 10), replace=False)
        shift = rng.normal(0.0, 2.0, size=len(cols))
        X_obs[i, cols] = np.nan_to_num(X_obs[i, cols], nan=0.0) + shift

    # (b) Scale a whole template block
    for i in out_idx[n_out // 4: n_out // 2]:
        tname, _ = templates[rng.integers(0, len(templates))]
        cols = np.where(feature_template == tname)[0]
        factor = rng.uniform(2.0, 5.0)
        X_obs[i, cols] = np.nan_to_num(X_obs[i, cols], nan=0.0) * factor

    # (c) Block anomaly across 2 templates (correlation break)
    for i in out_idx[n_out // 2: 3 * n_out // 4]:
        t1, _ = templates[rng.integers(0, len(templates))]
        t2, _ = templates[rng.integers(0, len(templates))]
        cols = np.r_[np.where(feature_template == t1)[0], np.where(feature_template == t2)[0]]
        X_obs[i, cols] = rng.normal(0.0, 1.2, size=len(cols))

    # (d) Sign flip on a subset (economically questionable, good stress test)
    for i in out_idx[3 * n_out // 4:]:
        cols = rng.choice(n_features, size=rng.integers(5, 15), replace=False)
        X_obs[i, cols] = -np.nan_to_num(X_obs[i, cols], nan=0.0)

    return X_obs, y, feature_template


# ----------------------------
# 2) Simple preprocessing: scale + missing sentinel
# ----------------------------
def preprocess_for_vae(X, sentinel=-1.0):
    # Fit scaler on non-missing only (simple but acceptable for simulation)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_filled = np.nan_to_num(X, nan=np.nan)  # keep nan for now
    # scaler can't handle nan, so fit on observed rows per feature
    # approach: fit on nan-filled with feature-wise median
    X_med = X.copy()
    med = np.nanmedian(X_med, axis=0)
    inds = np.where(np.isnan(X_med))
    X_med[inds] = np.take(med, inds[1])
    X_scaled = scaler.fit_transform(X_med)

    # After scaling, put missing back as sentinel
    X_out = X_scaled.copy()
    X_out[np.isnan(X)] = sentinel
    return X_out, scaler


# ----------------------------
# 3) Train/evaluate VAE
# ----------------------------
X, y, feature_template = simulate_finrep_like()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

X_train_p, scaler = preprocess_for_vae(X_train, sentinel=-1.0)
X_test_p, _ = preprocess_for_vae(X_test, sentinel=-1.0)  # for real use: transform with same scaler

# IMPORTANT: In real pipeline, do NOT refit scaler on test; here it’s a demo.
# apply scaler.transform on median-imputed test for strictness

vae = VAE(
    contamination=0.04,        # match outlier_frac
    encoder_neurons=[128, 64],
    decoder_neurons=[64, 128],
    latent_dim=16,
    epochs=30,
    batch_size=256,
    learning_rate=1e-3,
    verbose=0,
)

vae.fit(X_train_p)

scores = vae.decision_function(X_test_p)  # higher = more outlier
auc = roc_auc_score(y_test, scores)
ap = average_precision_score(y_test, scores)

print(f"AUROC: {auc:.3f}")
print(f"Avg Precision: {ap:.3f}")

# ----------------------------
# 4) Minimal explainability: top features by reconstruction error (approx)
# ----------------------------
# pyod VAE exposes reconstruction errors via internal torch model only indirectly.
# alternative: use per-feature deviation from median of training as a proxy,
# OR (better) keep torch VAE and compute per-feature recon error.
#
# For now: proxy "feature contribution" = abs(x - train_median) ignoring missing sentinel.
train_median = np.median(np.where(X_train_p > -0.9, X_train_p, np.nan), axis=0)
test_abs_dev = np.abs(np.where(X_test_p > -0.9, X_test_p, np.nan) - train_median)

top_n = 10
top_outliers = np.argsort(-scores)[:5]
for idx in top_outliers:
    contrib = test_abs_dev[idx]
    feat_ids = np.argsort(-np.nan_to_num(contrib, nan=-np.inf))[:top_n]
    print("\nSample", idx, "score", float(scores[idx]), "label", int(y_test[idx]))
    for j in feat_ids:
        print(f"  feat {j:3d} (template {feature_template[j]}): dev={contrib[j]:.3f}")

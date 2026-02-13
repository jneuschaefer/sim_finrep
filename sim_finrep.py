import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from pyod.models.vae import VAE

def simulate_finrep_like(
    n_samples=12000,
    templates=(("F01", 60), ("F02", 40), ("F18", 30), ("F20", 20)),
    n_latent=8,
    missing_rate=0.15,
    outlier_frac=0.04,
    seed=7,
):
    rng = np.random.default_rng(seed)

    feature_template = []
    for tname, k in templates:
        feature_template += [tname] * k
    feature_template = np.array(feature_template)
    n_features = len(feature_template)

    Z = rng.standard_normal((n_samples, n_latent))

    W = np.zeros((n_latent, n_features))
    col = 0
    for (tname, k) in templates:
        dom = rng.choice(n_latent, size=min(3, n_latent), replace=False)
        base = rng.normal(0.0, 0.15, size=(n_latent, k))
        base[dom, :] += rng.normal(0.0, 0.8, size=(len(dom), k))
        W[:, col:col + k] = base
        col += k

    X = Z @ W + rng.normal(0.0, 0.35, size=(n_samples, n_features))
    ratio_idx = rng.choice(n_features, size=int(0.25 * n_features), replace=False)
    X[:, ratio_idx] = np.tanh(X[:, ratio_idx])

    # structural + random missingness
    miss = np.zeros_like(X, dtype=bool)
    for (tname, _k) in templates:
        cols = np.where(feature_template == tname)[0]
        rows = rng.choice(n_samples, size=int(0.10 * n_samples), replace=False)
        miss[np.ix_(rows, cols)] = True
    miss |= (rng.random(X.shape) < missing_rate)

    X_obs = X.copy()
    X_obs[miss] = np.nan

    # labels
    y = np.zeros(n_samples, dtype=int)
    n_out = int(outlier_frac * n_samples)
    out_idx = rng.choice(n_samples, size=n_out, replace=False)
    y[out_idx] = 1

    # helper: pick only observed cols for a given row from a candidate set
    def observed_cols(row, cols):
        cols = np.asarray(cols)
        ok = ~np.isnan(X_obs[row, cols])
        return cols[ok]

    # (a) Shift some observed features
    for i in out_idx[: n_out // 4]:
        cols = rng.choice(n_features, size=rng.integers(3, 10), replace=False)
        cols = observed_cols(i, cols)
        if len(cols) == 0:
            continue
        shift = rng.normal(0.0, 2.0, size=len(cols))
        X_obs[i, cols] = X_obs[i, cols] + shift

    # (b) Scale a whole template block (only observed)
    for i in out_idx[n_out // 4: n_out // 2]:
        tname, _ = templates[rng.integers(0, len(templates))]
        cols = np.where(feature_template == tname)[0]
        cols = observed_cols(i, cols)
        if len(cols) == 0:
            continue
        factor = rng.uniform(2.0, 5.0)
        X_obs[i, cols] = X_obs[i, cols] * factor

    # (c) Break correlations in 2 templates (only observed)
    for i in out_idx[n_out // 2: 3 * n_out // 4]:
        t1, _ = templates[rng.integers(0, len(templates))]
        t2, _ = templates[rng.integers(0, len(templates))]
        cols = np.r_[np.where(feature_template == t1)[0], np.where(feature_template == t2)[0]]
        cols = observed_cols(i, cols)
        if len(cols) == 0:
            continue
        X_obs[i, cols] = rng.normal(0.0, 1.2, size=len(cols))

    # (d) Sign flip (only observed)
    for i in out_idx[3 * n_out // 4:]:
        cols = rng.choice(n_features, size=rng.integers(5, 15), replace=False)
        cols = observed_cols(i, cols)
        if len(cols) == 0:
            continue
        X_obs[i, cols] = -X_obs[i, cols]

    return X_obs, y, feature_template


def fit_transform_minmax_with_median_impute(X_train, X_test):
    # compute train medians for imputation
    train_med = np.nanmedian(X_train, axis=0)

    Xtr = X_train.copy()
    Xte = X_test.copy()

    # impute with train medians
    inds = np.where(np.isnan(Xtr)); Xtr[inds] = np.take(train_med, inds[1])
    inds = np.where(np.isnan(Xte)); Xte[inds] = np.take(train_med, inds[1])

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    return Xtr_s, Xte_s, train_med, scaler


# ---- run ----
X, y, feature_template = simulate_finrep_like()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42, stratify=y
)

X_train_p, X_test_p, train_med, scaler = fit_transform_minmax_with_median_impute(X_train, X_test)

vae = VAE(
    contamination=0.04,
    lr=1e-3,
    epoch_num=30,
    batch_size=256,
    latent_dim=16,
    encoder_neuron_list=[128, 64],
    decoder_neuron_list=[64, 128],
    verbose=1,
)

vae.fit(X_train_p)
scores = vae.decision_function(X_test_p)

auc = roc_auc_score(y_test, scores)
ap = average_precision_score(y_test, scores)

print(f"AUROC: {auc:.3f}")
print(f"Avg Precision: {ap:.3f}")

# Simple explain proxy: deviation from train median AFTER scaling
train_median_scaled = np.median(X_train_p, axis=0)
test_abs_dev = np.abs(X_test_p - train_median_scaled)

top_outliers = np.argsort(-scores)[:5]
top_n = 10
for idx in top_outliers:
    contrib = test_abs_dev[idx]
    feat_ids = np.argsort(-contrib)[:top_n]
    print("\nSample", idx, "score", float(scores[idx]), "label", int(y_test[idx]))
    for j in feat_ids:
        print(f"  feat {j:3d} (template {feature_template[j]}): dev={contrib[j]:.3f}")

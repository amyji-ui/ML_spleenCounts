import numpy as np

def kfold_cross_val(clf_factory, X, y, k=5, seed=42):
    X, y = np.asarray(X, float), np.asarray(y)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)

    class_indices = {c: rng.permutation(np.where(y == c)[0]) for c in classes}
    folds = [[] for _ in range(k)]
    for indices in class_indices.values():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)
    
    fold_scores = []
    for fold_idx in range(k):
        val_idx = np.array(folds[fold_idx])
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / sigma
        X_val = (X_val - mu) / sigma

        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        fold_scores.append(clf.score(X_val, y_val))

    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    return fold_scores, mean, std

def run_kfold_cross_val(clf_factory, X_train, y_train, k=5, seed=42, n_cv_folds=5):
    cv_scores = cv_mean = cv_std = None
    if n_cv_folds > 1:
        cv_scores, cv_mean, cv_std = kfold_cross_val(
            clf_factory,
            X=X_train, y=y_train, k=n_cv_folds, seed=seed
        )

    return dict(
        cv_scores=cv_scores,
        cv_mean=cv_mean,
        cv_std=cv_std
    )
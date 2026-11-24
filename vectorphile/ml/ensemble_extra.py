"""Extra ensemble methods — pure Python."""
import math
import random as _random
from typing import Optional, List

from vectorphile.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _uniq_sorted(seq):
    return sorted(set(seq))


class BaggingClassifier:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 random_state=None, n_jobs=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state

    def _make_estimator(self):
        if self.estimator is None:
            return DecisionTreeClassifier()
        import copy
        return copy.deepcopy(self.estimator)

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        n, p = len(X2), len(X2[0])
        self.classes_ = _uniq_sorted(y2)
        rng = _random.Random(self.random_state)
        n_samples = int(n * self.max_samples) if isinstance(self.max_samples, float) else self.max_samples
        self.estimators_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idxs = [rng.randint(0, n - 1) for _ in range(n_samples)]
            else:
                idxs = rng.sample(range(n), min(n_samples, n))
            Xb = [X2[i] for i in idxs]
            yb = [y2[i] for i in idxs]
            est = self._make_estimator()
            est.fit(Xb, yb)
            self.estimators_.append(est)
        return self

    def predict_proba(self, X):
        K = len(self.classes_)
        avg = [[0.0] * K for _ in range(len(X))]
        count = 0
        for est in self.estimators_:
            if hasattr(est, 'predict_proba'):
                proba = est.predict_proba(X)
                for i, row in enumerate(proba):
                    # align to self.classes_
                    est_classes = getattr(est, 'classes_', self.classes_)
                    for ci, c in enumerate(est_classes):
                        if c in self.classes_:
                            k = self.classes_.index(c)
                            avg[i][k] += row[ci]
            else:
                preds = est.predict(X)
                for i, pred in enumerate(preds):
                    if pred in self.classes_:
                        k = self.classes_.index(pred)
                        avg[i][k] += 1.0
            count += 1
        T = max(count, 1)
        return [[v / T for v in row] for row in avg]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class BaggingRegressor:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 bootstrap=True, random_state=None, n_jobs=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _make_estimator(self):
        if self.estimator is None:
            return DecisionTreeRegressor()
        import copy
        return copy.deepcopy(self.estimator)

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(X2)
        rng = _random.Random(self.random_state)
        n_samples = int(n * self.max_samples) if isinstance(self.max_samples, float) else self.max_samples
        self.estimators_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idxs = [rng.randint(0, n - 1) for _ in range(n_samples)]
            else:
                idxs = rng.sample(range(n), min(n_samples, n))
            Xb = [X2[i] for i in idxs]
            yb = [y2[i] for i in idxs]
            est = self._make_estimator()
            est.fit(Xb, yb)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        all_preds = [est.predict(X) for est in self.estimators_]
        n = len(X)
        return [sum(all_preds[t][i] for t in range(len(self.estimators_))) / len(self.estimators_)
                for i in range(n)]

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class AdaBoostClassifier:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _make_estimator(self):
        if self.estimator is None:
            return DecisionTreeClassifier(max_depth=1)
        import copy
        return copy.deepcopy(self.estimator)

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        n = len(X2)
        self.classes_ = _uniq_sorted(y2)
        rng = _random.Random(self.random_state)
        weights = [1.0 / n] * n
        self.estimators_ = []
        self.estimator_weights_ = []
        for _ in range(self.n_estimators):
            # Weighted sample
            cumulative = []
            cum = 0.0
            for w in weights:
                cum += w
                cumulative.append(cum)
            idxs = []
            for _ in range(n):
                r = rng.random()
                for idx, c in enumerate(cumulative):
                    if r <= c:
                        idxs.append(idx)
                        break
                else:
                    idxs.append(n - 1)
            Xb = [X2[i] for i in idxs]
            yb = [y2[i] for i in idxs]
            est = self._make_estimator()
            est.fit(Xb, yb)
            preds = est.predict(X2)
            # Compute weighted error
            err = sum(weights[i] for i in range(n) if preds[i] != y2[i])
            err = max(err, 1e-10)
            if err >= 0.5:
                break
            alpha = self.learning_rate * 0.5 * math.log((1 - err) / err)
            # Update weights
            for i in range(n):
                if preds[i] == y2[i]:
                    weights[i] *= math.exp(-alpha)
                else:
                    weights[i] *= math.exp(alpha)
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
        return self

    def predict(self, X):
        K = len(self.classes_)
        scores = [[0.0] * K for _ in range(len(X))]
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = est.predict(X)
            for i, pred in enumerate(preds):
                if pred in self.classes_:
                    k = self.classes_.index(pred)
                    scores[i][k] += alpha
        return [self.classes_[row.index(max(row))] for row in scores]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class VotingClassifier:
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = _uniq_sorted(y)
        self.estimators_ = []
        for name, est in self.estimators:
            import copy
            cloned = copy.deepcopy(est)
            cloned.fit(X, y)
            self.estimators_.append(cloned)
        return self

    def predict(self, X):
        K = len(self.classes_)
        weights = self.weights or [1.0] * len(self.estimators_)
        if self.voting == 'soft':
            avg = [[0.0] * K for _ in range(len(X))]
            for est, w in zip(self.estimators_, weights):
                proba = est.predict_proba(X)
                for i, row in enumerate(proba):
                    for k in range(K):
                        avg[i][k] += w * row[k]
            return [self.classes_[row.index(max(row))] for row in avg]
        # hard voting
        result = []
        all_preds = [est.predict(X) for est in self.estimators_]
        for i in range(len(X)):
            votes = {}
            for t, preds in enumerate(all_preds):
                c = preds[i]
                votes[c] = votes.get(c, 0) + weights[t]
            result.append(max(votes, key=votes.get))
        return result

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class StackingClassifier:
    def __init__(self, estimators, final_estimator=None, cv=5, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    def fit(self, X, y):
        from vectorphile.ml.model_selection import KFold
        import copy
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = _uniq_sorted(y2)
        n = len(X2)
        T = len(self.estimators)
        meta_X = [[0.0] * T for _ in range(n)]
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        # Train base estimators with cross-val predictions for meta-features
        self.estimators_ = []
        for t, (name, est) in enumerate(self.estimators):
            cloned = copy.deepcopy(est)
            for train_idx, test_idx in kf.split(X2):
                Xtr = [X2[i] for i in train_idx]
                ytr = [y2[i] for i in train_idx]
                Xte = [X2[i] for i in test_idx]
                e = copy.deepcopy(est)
                e.fit(Xtr, ytr)
                preds = e.predict(Xte)
                for i, idx in enumerate(test_idx):
                    meta_X[idx][t] = float(preds[i]) if isinstance(preds[i], (int, float)) else self.classes_.index(preds[i])
            cloned.fit(X2, y2)
            self.estimators_.append(cloned)
        if self.passthrough:
            meta_X = [X2[i] + meta_X[i] for i in range(n)]
        if self.final_estimator is None:
            from vectorphile.ml.linear import LogisticRegression
            self.final_estimator_ = LogisticRegression()
        else:
            self.final_estimator_ = copy.deepcopy(self.final_estimator)
        self.final_estimator_.fit(meta_X, y2)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        T = len(self.estimators_)
        meta_X = [[0.0] * T for _ in range(len(X2))]
        for t, est in enumerate(self.estimators_):
            preds = est.predict(X2)
            for i, pred in enumerate(preds):
                meta_X[i][t] = float(pred) if isinstance(pred, (int, float)) else self.classes_.index(pred)
        if self.passthrough:
            meta_X = [X2[i] + meta_X[i] for i in range(len(X2))]
        return self.final_estimator_.predict(meta_X)

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class AdaBoostRegressor:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 loss='linear', random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def _make_estimator(self):
        if self.estimator is None:
            return DecisionTreeRegressor(max_depth=3)
        import copy
        return copy.deepcopy(self.estimator)

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(X2)
        rng = _random.Random(self.random_state)
        weights = [1.0 / n] * n
        self.estimators_ = []
        self.estimator_weights_ = []
        for _ in range(self.n_estimators):
            cumulative, cum = [], 0.0
            for w in weights:
                cum += w
                cumulative.append(cum)
            idxs = []
            for _ in range(n):
                r = rng.random()
                for idx, c in enumerate(cumulative):
                    if r <= c:
                        idxs.append(idx)
                        break
                else:
                    idxs.append(n - 1)
            est = self._make_estimator()
            est.fit([X2[i] for i in idxs], [y2[i] for i in idxs])
            preds = est.predict(X2)
            errors = [abs(preds[i] - y2[i]) for i in range(n)]
            max_err = max(errors) + 1e-10
            norm_errors = [e / max_err for e in errors]
            loss_val = min(sum(weights[i] * norm_errors[i] for i in range(n)), 1.0 - 1e-10)
            if loss_val >= 0.5:
                break
            beta = loss_val / (1.0 - loss_val + 1e-10)
            alpha = self.learning_rate * math.log(1.0 / (beta + 1e-10))
            weights = [weights[i] * beta ** (1.0 - norm_errors[i]) for i in range(n)]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
        return self

    def predict(self, X):
        if not self.estimators_:
            return [0.0] * len(X)
        preds_all = [est.predict(X) for est in self.estimators_]
        total_w = sum(self.estimator_weights_)
        return [sum(self.estimator_weights_[t] * preds_all[t][i]
                    for t in range(len(self.estimators_))) / max(total_w, 1e-10)
                for i in range(len(X))]

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class ExtraTreesClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = _uniq_sorted(y2)
        n = len(X2)
        rng = _random.Random(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idxs = [rng.randint(0, n - 1) for _ in range(n)]
            est = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            est.fit([X2[i] for i in idxs], [y2[i] for i in idxs])
            self.estimators_.append(est)
        return self

    def predict(self, X):
        from collections import Counter
        votes = [est.predict(X) for est in self.estimators_]
        return [Counter(votes[t][i] for t in range(len(self.estimators_))).most_common(1)[0][0]
                for i in range(len(X))]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class ExtraTreesRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(X2)
        rng = _random.Random(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idxs = [rng.randint(0, n - 1) for _ in range(n)]
            est = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            est.fit([X2[i] for i in idxs], [y2[i] for i in idxs])
            self.estimators_.append(est)
        return self

    def predict(self, X):
        preds_all = [est.predict(X) for est in self.estimators_]
        return [sum(preds_all[t][i] for t in range(len(self.estimators_))) / len(self.estimators_)
                for i in range(len(X))]

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class VotingRegressor:
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        import copy
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        self.estimators_ = [copy.deepcopy(est).fit(X2, y2) for _, est in self.estimators]
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        weights = self.weights or [1.0] * len(self.estimators_)
        total_w = sum(weights)
        preds_all = [est.predict(X2) for est in self.estimators_]
        return [sum(weights[t] * preds_all[t][i] for t in range(len(self.estimators_))) / total_w
                for i in range(len(X2))]

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class StackingRegressor:
    def __init__(self, estimators, final_estimator=None, cv=5, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    def fit(self, X, y):
        from vectorphile.ml.model_selection import KFold
        import copy
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(X2)
        T = len(self.estimators)
        meta_X = [[0.0] * T for _ in range(n)]
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        self.estimators_ = []
        for t, (name, est) in enumerate(self.estimators):
            cloned = copy.deepcopy(est)
            for train_idx, test_idx in kf.split(X2):
                Xtr = [X2[i] for i in train_idx]
                ytr = [y2[i] for i in train_idx]
                Xte = [X2[i] for i in test_idx]
                e = copy.deepcopy(est)
                e.fit(Xtr, ytr)
                preds = e.predict(Xte)
                for i, idx in enumerate(test_idx):
                    meta_X[idx][t] = float(preds[i])
            cloned.fit(X2, y2)
            self.estimators_.append(cloned)
        if self.passthrough:
            meta_X = [X2[i] + meta_X[i] for i in range(n)]
        if self.final_estimator is None:
            from vectorphile.ml.linear import LinearRegression
            self.final_estimator_ = LinearRegression()
        else:
            self.final_estimator_ = copy.deepcopy(self.final_estimator)
        self.final_estimator_.fit(meta_X, y2)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        T = len(self.estimators_)
        meta_X = [[0.0] * T for _ in range(len(X2))]
        for t, est in enumerate(self.estimators_):
            preds = est.predict(X2)
            for i, pred in enumerate(preds):
                meta_X[i][t] = float(pred)
        if self.passthrough:
            meta_X = [X2[i] + meta_X[i] for i in range(len(X2))]
        return self.final_estimator_.predict(meta_X)

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))

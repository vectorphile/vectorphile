"""Tests for ML algorithms."""
import numpy as np
import pytest
import vectorphile as vp
from vectorphile import ml


def _make_classification(n=200, n_features=4, n_classes=2, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    if n_classes > 2:
        y = (X[:, 0] * 3).astype(int) % n_classes
    return X, y


def _make_regression(n=200, n_features=4, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, n_features)
    coef = np.array([1.0, -2.0, 0.5, 1.5])
    y = X @ coef + 0.1 * np.random.randn(n)
    return X, y


class TestLinearModels:
    def test_linear_regression(self):
        X, y = _make_regression()
        model = ml.LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.9

    def test_ridge(self):
        X, y = _make_regression()
        model = ml.Ridge(alpha=0.1)
        model.fit(X, y)
        assert model.score(X, y) > 0.9

    def test_lasso(self):
        X, y = _make_regression()
        model = ml.Lasso(alpha=0.01)
        model.fit(X, y)
        assert model.score(X, y) > 0.8

    def test_logistic_regression(self):
        X, y = _make_classification()
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2, random_state=42)
        model = ml.LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        assert acc > 0.75

    def test_logistic_multiclass(self):
        X, y = _make_classification(n_classes=3)
        model = ml.LogisticRegression(max_iter=100)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(np.unique(preds)) >= 2


class TestTreeModels:
    def test_dt_classifier(self):
        X, y = _make_classification()
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2)
        model = ml.DecisionTreeClassifier(max_depth=5)
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.8

    def test_dt_regressor(self):
        X, y = _make_regression()
        model = ml.DecisionTreeRegressor(max_depth=5)
        model.fit(X, y)
        assert model.score(X, y) > 0.8

    def test_random_forest_classifier(self):
        X, y = _make_classification()
        model = ml.RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert model.score(X, y) > 0.9

    def test_random_forest_regressor(self):
        X, y = _make_regression()
        model = ml.RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert model.score(X, y) > 0.9


class TestClustering:
    def test_kmeans(self):
        np.random.seed(42)
        X = np.vstack([np.random.randn(50, 2) + c for c in [(0, 0), (5, 5), (-5, 5)]])
        model = ml.KMeans(n_clusters=3, random_state=42)
        labels = model.fit_predict(X)
        assert len(np.unique(labels)) == 3

    def test_dbscan(self):
        np.random.seed(42)
        X = np.vstack([np.random.randn(30, 2) + c for c in [(0, 0), (10, 10)]])
        model = ml.DBSCAN(eps=1.5, min_samples=3)
        labels = model.fit_predict(X)
        # Should find at least 2 clusters (labels >= 0)
        assert (labels >= 0).sum() > 40


class TestDecomposition:
    def test_pca(self):
        X = np.random.randn(100, 10)
        pca = ml.PCA(n_components=3)
        X_transformed = pca.fit_transform(X)
        assert X_transformed.shape == (100, 3)
        # Explained variance should sum <= 1
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-5

    def test_nmf(self):
        X = np.abs(np.random.randn(50, 8))
        nmf = ml.NMF(n_components=3, random_state=42)
        W = nmf.fit_transform(X)
        assert W.shape == (50, 3)


class TestNeighbors:
    def test_knn_classifier(self):
        X, y = _make_classification()
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2)
        model = ml.KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.8

    def test_knn_regressor(self):
        X, y = _make_regression()
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2)
        model = ml.KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.5


class TestNaiveBayes:
    def test_gaussian_nb(self):
        X, y = _make_classification()
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2)
        model = ml.GaussianNB()
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.75


class TestPreprocessing:
    def test_standard_scaler(self):
        X = np.random.randn(100, 4) * 10 + 5
        scaler = ml.StandardScaler()
        Xt = scaler.fit_transform(X)
        assert abs(Xt.mean()) < 0.1
        assert abs(Xt.std() - 1.0) < 0.1

    def test_min_max_scaler(self):
        X = np.random.randn(100, 4)
        scaler = ml.MinMaxScaler()
        Xt = scaler.fit_transform(X)
        assert Xt.min() >= 0.0 - 1e-5
        assert Xt.max() <= 1.0 + 1e-5

    def test_label_encoder(self):
        le = ml.LabelEncoder()
        y = ['cat', 'dog', 'cat', 'bird']
        encoded = le.fit_transform(y)
        assert len(np.unique(encoded)) == 3
        assert list(le.inverse_transform(encoded)) == y

    def test_one_hot(self):
        ohe = ml.OneHotEncoder()
        X = np.array(['a', 'b', 'c', 'a']).reshape(-1, 1)
        result = ohe.fit_transform(X)
        assert result.shape == (4, 3)
        assert result.sum() == 4


class TestMetrics:
    def test_accuracy(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        assert ml.accuracy_score(y_true, y_pred) == 0.75

    def test_f1(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 1, 0]
        assert ml.f1_score(y_true, y_pred) == 1.0

    def test_mse(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        assert ml.mse(y_true, y_pred) == 0.0

    def test_r2(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        assert abs(ml.r2_score(y_true, y_pred) - 1.0) < 1e-10

    def test_confusion_matrix(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 1, 1]
        cm = ml.confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm.sum() == 4


class TestModelSelection:
    def test_train_test_split(self):
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_kfold(self):
        X = np.random.randn(100, 4)
        kf = ml.KFold(n_splits=5)
        folds = list(kf.split(X))
        assert len(folds) == 5
        total_test = sum(len(test) for _, test in folds)
        assert total_test == 100

    def test_cross_val_score(self):
        X, y = _make_classification()
        model = ml.LogisticRegression(max_iter=100)
        scores = ml.cross_val_score(model, X, y, cv=3)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

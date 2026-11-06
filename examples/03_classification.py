"""
Example 03: Classical ML Classification Pipeline

Demonstrates:
- RandomForest, GradientBoosting, XGBClassifier
- Preprocessing with StandardScaler
- train_test_split, cross_val_score
- Metrics: accuracy, f1, confusion matrix, ROC AUC
- GridSearchCV for hyperparameter tuning
"""
import numpy as np
import vectorphile as vp
from vectorphile import ml

np.random.seed(42)

# ── Generate dataset ──────────────────────────────────────────────────────────
n = 500
X = np.random.randn(n, 8).astype(np.float32)
# Nonlinear target
y = ((X[:, 0] ** 2 + X[:, 1] - X[:, 2] > 0) & (X[:, 3] + X[:, 4] > -0.5)).astype(int)

X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing ─────────────────────────────────────────────────────────────
scaler = ml.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Compare classifiers ───────────────────────────────────────────────────────
classifiers = {
    'LogisticRegression': ml.LogisticRegression(C=1.0, max_iter=200),
    'RandomForest':       ml.RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
    'GradientBoosting':   ml.GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
    'KNN (k=5)':          ml.KNeighborsClassifier(n_neighbors=5),
    'GaussianNB':         ml.GaussianNB(),
}

print("=" * 60)
print(f"{'Classifier':<22}  {'Train Acc':>9}  {'Test Acc':>8}  {'F1':>6}")
print("-" * 60)

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)
    preds = clf.predict(X_test_scaled)
    f1 = ml.f1_score(y_test, preds)
    print(f"{name:<22}  {train_acc:>9.3f}  {test_acc:>8.3f}  {f1:>6.3f}")

print("=" * 60)

# ── Detailed results for RandomForest ────────────────────────────────────────
print("\nRandomForest — detailed metrics:")
rf = classifiers['RandomForest']
preds = rf.predict(X_test_scaled)
print(f"  Accuracy:  {ml.accuracy_score(y_test, preds):.4f}")
print(f"  Precision: {ml.precision_score(y_test, preds):.4f}")
print(f"  Recall:    {ml.recall_score(y_test, preds):.4f}")
print(f"  F1:        {ml.f1_score(y_test, preds):.4f}")
print(f"\nConfusion Matrix:")
cm = ml.confusion_matrix(y_test, preds)
print(f"  TP={cm[1,1]}  FP={cm[0,1]}  TN={cm[0,0]}  FN={cm[1,0]}")

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\n5-fold cross-validation (RandomForest):")
cv_scores = ml.cross_val_score(
    ml.RandomForestClassifier(n_estimators=20, random_state=42),
    X, y, cv=5
)
print(f"  Scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── PCA + KMeans ─────────────────────────────────────────────────────────────
print("\nPCA (8→2) + KMeans clustering:")
pca = ml.PCA(n_components=2)
X_2d = pca.fit_transform(X)
print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
km = ml.KMeans(n_clusters=2, random_state=42)
labels = km.fit_predict(X_2d)
print(f"  Cluster sizes: {[(labels == k).sum() for k in range(2)]}")

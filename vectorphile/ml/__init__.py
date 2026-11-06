from vectorphile.ml import linear, tree, ensemble, cluster, decomposition, neighbors, naive_bayes, preprocessing, metrics, model_selection, svm
from vectorphile.ml.linear import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, Perceptron
from vectorphile.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from vectorphile.ml.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from vectorphile.ml.cluster import KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture
from vectorphile.ml.decomposition import PCA, TruncatedSVD, NMF, FastICA
from vectorphile.ml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from vectorphile.ml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from vectorphile.ml.svm import SVC, SVR
from vectorphile.ml.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, PolynomialFeatures,
)
from vectorphile.ml.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, mse, rmse, mae, r2_score, silhouette_score, log_loss,
)
from vectorphile.ml.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV,
)
from vectorphile.ml.boost import XGBClassifier, XGBRegressor

__all__ = [
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LogisticRegression', 'Perceptron',
    'DecisionTreeClassifier', 'DecisionTreeRegressor',
    'RandomForestClassifier', 'RandomForestRegressor',
    'GradientBoostingClassifier', 'GradientBoostingRegressor',
    'XGBClassifier', 'XGBRegressor',
    'KMeans', 'DBSCAN', 'AgglomerativeClustering', 'GaussianMixture',
    'PCA', 'TruncatedSVD', 'NMF', 'FastICA',
    'KNeighborsClassifier', 'KNeighborsRegressor',
    'GaussianNB', 'MultinomialNB', 'BernoulliNB',
    'SVC', 'SVR',
    'StandardScaler', 'MinMaxScaler', 'RobustScaler',
    'LabelEncoder', 'OneHotEncoder', 'PolynomialFeatures',
    'accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score',
    'confusion_matrix', 'mse', 'rmse', 'mae', 'r2_score', 'silhouette_score', 'log_loss',
    'train_test_split', 'KFold', 'StratifiedKFold', 'cross_val_score',
    'GridSearchCV', 'RandomizedSearchCV',
]

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
from vectorphile.ml.lgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from vectorphile.ml.catboost_impl import CatBoostClassifier, CatBoostRegressor, CatBoostRanker
from vectorphile.ml.anomaly import IsolationForest, LocalOutlierFactor, EllipticEnvelope, OneClassSVM
from vectorphile.ml.ensemble_extra import (
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
)
from vectorphile.ml.pipeline import Pipeline, ColumnTransformer, FeatureUnion, make_pipeline, make_union
from vectorphile.ml.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile, GenericUnivariateSelect,
    SelectFpr, SelectFdr, SelectFwe, SelectFromModel, RFE, RFECV,
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
)
from vectorphile.ml.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding, UMAP

__all__ = [
    # Linear
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LogisticRegression', 'Perceptron',
    # Trees
    'DecisionTreeClassifier', 'DecisionTreeRegressor',
    # Ensembles
    'RandomForestClassifier', 'RandomForestRegressor',
    'GradientBoostingClassifier', 'GradientBoostingRegressor',
    'XGBClassifier', 'XGBRegressor',
    'LGBMClassifier', 'LGBMRegressor', 'LGBMRanker',
    'CatBoostClassifier', 'CatBoostRegressor', 'CatBoostRanker',
    'BaggingClassifier', 'BaggingRegressor',
    'AdaBoostClassifier', 'AdaBoostRegressor',
    'ExtraTreesClassifier', 'ExtraTreesRegressor',
    'VotingClassifier', 'VotingRegressor',
    'StackingClassifier', 'StackingRegressor',
    # Clustering
    'KMeans', 'DBSCAN', 'AgglomerativeClustering', 'GaussianMixture',
    # Decomposition
    'PCA', 'TruncatedSVD', 'NMF', 'FastICA',
    # Neighbors
    'KNeighborsClassifier', 'KNeighborsRegressor',
    # Naive Bayes
    'GaussianNB', 'MultinomialNB', 'BernoulliNB',
    # SVM
    'SVC', 'SVR',
    # Anomaly detection
    'IsolationForest', 'LocalOutlierFactor', 'EllipticEnvelope', 'OneClassSVM',
    # Manifold
    'TSNE', 'MDS', 'Isomap', 'LocallyLinearEmbedding', 'SpectralEmbedding', 'UMAP',
    # Pipeline
    'Pipeline', 'ColumnTransformer', 'FeatureUnion', 'make_pipeline', 'make_union',
    # Feature selection
    'VarianceThreshold', 'SelectKBest', 'SelectPercentile', 'GenericUnivariateSelect',
    'SelectFpr', 'SelectFdr', 'SelectFwe', 'SelectFromModel', 'RFE', 'RFECV',
    'chi2', 'f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression',
    # Preprocessing
    'StandardScaler', 'MinMaxScaler', 'RobustScaler',
    'LabelEncoder', 'OneHotEncoder', 'PolynomialFeatures',
    # Metrics
    'accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score',
    'confusion_matrix', 'mse', 'rmse', 'mae', 'r2_score', 'silhouette_score', 'log_loss',
    # Model selection
    'train_test_split', 'KFold', 'StratifiedKFold', 'cross_val_score',
    'GridSearchCV', 'RandomizedSearchCV',
]

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.linear_model import LinearRegression

import numpy as np

my_models = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'XGBoost': XGBRegressor(),
    'CatBoost': CatBoostRegressor(),
    'LGBM': LGBMRegressor(),
    'KN-Neighbors': KNeighborsRegressor(),
    'Multiple Linear': LinearRegression()
}

model_params = {
    'Random Forest': {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=50)],
        'max_features': [1, 2, None, 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(3, 20, num=18)],
        'min_samples_split': [int(x) for x in np.linspace(start=10, stop=50, num=5)],
        'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=10, num=10)],
        'max_leaf_nodes': [int(x) for x in np.linspace(start=5, stop=20, num=5)],
        'max_samples': [float(x) for x in np.linspace(start=0.3, stop=0.7, num=5)],
        'bootstrap': [True],
        'oob_score': [True],
        'criterion': ["squared_error"],
        'ccp_alpha': [float(x) for x in np.linspace(start=0.0, stop=0.4, num=10)],
        'random_state': [100]
    },
    'Gradient Boosting': {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=50)],
        'max_features': [1, 2, None, 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(3, 20, num=18)],
        'min_samples_split': [int(x) for x in np.linspace(start=10, stop=50, num=5)],
        'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=10, num=10)],
        'max_leaf_nodes': [int(x) for x in np.linspace(start=5, stop=20, num=5)],
        'subsample': [float(x) for x in np.linspace(start=0.3, stop=0.7, num=5)],
        'learning_rate': [float(x) for x in np.linspace(start=0.01, stop=0.5, num=10)],
        'criterion': ['squared_error'],
        'loss': ['squared_error', 'huber'],
        'ccp_alpha': [float(x) for x in np.linspace(start=0.0, stop=0.4, num=10)],
        'random_state': [100]
    },
    'Extra Trees': {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=10000, num=50)],
        'max_features': [1, 2, None, 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(3, 20, num=18)],
        'min_samples_split': [int(x) for x in np.linspace(start=10, stop=50, num=5)],
        'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=10, num=10)],
        'max_leaf_nodes': [int(x) for x in np.linspace(start=5, stop=20, num=5)],
        'max_samples': [float(x) for x in np.linspace(start=0.3, stop=0.7, num=5)],
        'bootstrap': [True],
        'oob_score': [True],
        'criterion': ["squared_error"],
        'ccp_alpha': [float(x) for x in np.linspace(start=0.0, stop=1, num=10)],
        'random_state': [100]
    },
    'XGBoost': {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=50)],
        'eta': [float(x) for x in np.linspace(start=0.01, stop=0.1, num=20)],
        # 'learning_rate': [float(x) for x in np.linspace(start=0.01, stop=0.1, num=10)],
        'max_depth': [int(x) for x in np.linspace(3, 20, num=18)],
        'subsample': [float(x) for x in np.linspace(start=0.3, stop=0.8, num=5)],
        'min_child_weight': [int(x) for x in np.linspace(start=1, stop=30, num=20)],
        'colsample_bytree': [float(x) for x in np.linspace(start=0.3, stop=0.8, num=5)],
        'gamma': [float(x) for x in np.linspace(start=0.1, stop=0.8, num=9)],
        # 'early_stopping_rounds': [int(x) for x in np.linspace(10, 20, num=10)],
        'seed': [100]
    },
    'CatBoost': {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=500, num=50)],
        'learning_rate': [float(x) for x in np.linspace(start=0.1, stop=0.8, num=10)],
        'subsample': [float(x) for x in np.linspace(start=0.3, stop=0.8, num=5)],
        'max_depth': [int(x) for x in np.linspace(1, 16, num=16)],
        'rsm': [float(x) for x in np.linspace(start=0.3, stop=0.8, num=6)],
        # 'bagging_temperature': [float(x) for x in np.linspace(start=0, stop=10, num=5)],
        # 'random_strength': [int(x) for x in np.linspace(start=0, stop=10, num=5)],
        # 'border_count': [int(x) for x in np.linspace(start=1, stop=10, num=5)],
        # 'fold_len_multplier': [float(x) for x in np.linspace(start=1, stop=10, num=10)],
        # 'approx_on_full_history': [True],
        'eval_metric': ['RMSE', 'R2'],
        'loss_function': ['RMSE'],
        'sampling_frequency': ['PerTree', 'PerTreeLevel'],
        'min_child_samples': [int(x) for x in np.linspace(start=5, stop=30, num=10)],
        'verbose': [False],
        # 'early_stopping_roundsearly_stopping_rounds': [True],
        'random_seed': [100]
    },
    'LGBM': {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=50)],
        'max_depth': [int(x) for x in np.linspace(3, 20, num=18)],
        'num_leaves': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
        'learning_rate': [float(x) for x in np.linspace(start=0.01, stop=0.1, num=10)],
        'subsample': [float(x) for x in np.linspace(start=0.3, stop=0.7, num=5)],
        'subsample_freq': [int(x) for x in np.linspace(start=10, stop=50, num=5)],
        'boosting_type': ['gbdt', 'dart', 'rf'],
        'colsample_bytree': [float(x) for x in np.linspace(start=0.3, stop=0.7, num=5)],
        'min_child_samples': [int(x) for x in np.linspace(start=1, stop=10, num=10)],
        'extra_tree': [True, False],
        'random_state': [100]
    },
    'KN-Neighbors': {
        'n_neighbors': [int(x) for x in np.linspace(start=1, stop=2, num=3)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [int(x) for x in np.linspace(start=3, stop=10, num=8)],
        'p': [int(x) for x in np.linspace(start=1, stop=10, num=10)],
    }
}
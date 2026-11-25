import numpy as np
import pandas as pd
import mlflow
from omegaconf import OmegaConf
from datetime import datetime
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data import (
    create_splits,
    fill_none,
    load_data,
    make_pipeline,
    prepare_data_by_folds
)
from src.env import seed_everything

from src.metrics import get_objective_metric, compute_metrics

import warnings
warnings.filterwarnings('ignore')

cfg = OmegaConf.load('config.yaml')

# DATA PREPARATION
df = load_data(cfg)
df = fill_none(df)
pipeline = make_pipeline(cfg, df) # linear pipeline in config
df = create_splits(cfg, df)

# Run training

model_dict = {
    'LinearRegression': LinearRegression(),
    'L1_Regression': Lasso(alpha=0.005),
    'L2_regression': Ridge(alpha=500),
    'ElasticNet': ElasticNet(alpha=1, l1_ratio=0.0005),
    'KNN': KNeighborsRegressor(n_neighbors=6),
    'DecisionTree': DecisionTreeRegressor(max_depth=6),
    'RandomForest': RandomForestRegressor(max_depth=10, n_estimators=200)}

# Run cycle
experiment_name = f"other_models_{datetime.now().strftime('%d%m%Y_%H%M')}"
mlflow.set_experiment(experiment_name)

for name, model in model_dict.items():
    with mlflow.start_run(run_name=f'{name}'):

        all_targets = []
        all_preds = [] # for oof rmsle

        #Folds cycle
        for fold in range(cfg.data.n_splits):
            seed_everything(cfg.general.seed + fold)
            # Data preparation
            X_train, y_train, X_val, y_val = prepare_data_by_folds(cfg, df, pipeline, fold)
            
            if cfg.data.log_transform_target:
                y_train = np.log1p(y_train)

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            # Filling lists
            if cfg.data.log_transform_target:
                preds = np.expm1(preds)

            all_targets.extend(y_val)
            all_preds.extend(preds)

        obj_name, obj_spec = get_objective_metric(cfg)
        fold_score = compute_metrics(
            all_targets, all_preds, {obj_name: obj_spec}, y_proba=None)[obj_name] #OOF RMSLE
        
        mlflow.log_metric(f'{obj_name}', fold_score)
        print(f'{name}_{obj_name} - {fold_score}')

    
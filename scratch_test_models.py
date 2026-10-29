import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

data_c = np.load('data/cached_features/features_conjunctiva.npz', allow_pickle=True)
X_c = data_c['features']
y_c = data_c['hemoglobin']
splits = data_c['splits']

train_mask = (splits == 'train') | (splits == 'val')
test_mask = (splits == 'test')

X_train, y_train = X_c[train_mask], y_c[train_mask]
X_test, y_test = X_c[test_mask], y_c[test_mask]

candidates = {
    'Ridge (alpha=300)': make_pipeline(StandardScaler(), Ridge(alpha=300.0)),
    'Ridge (alpha=100)': make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
    'BayesianRidge': make_pipeline(StandardScaler(), BayesianRidge()),
    'HuberRegressor': make_pipeline(StandardScaler(), HuberRegressor(alpha=50.0)),
    'SVR (RBF, C=1.0)': make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)),
    'SVR (RBF, C=2.0)': make_pipeline(StandardScaler(), SVR(C=2.0, epsilon=0.1)),
    'ElasticNet (l1=0.1)': make_pipeline(StandardScaler(), ElasticNet(alpha=0.5, l1_ratio=0.1)),
}

print(f"{'Model':<25} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'Pearson r':>10}")
print("-" * 65)

for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    corr, _ = stats.pearsonr(y_test, preds)
    print(f"{name:<25} {mae:>7.3f} {rmse:>7.3f} {r2:>7.4f} {corr:>10.4f}")

# Also test an ensemble / blend of Ridge + SVR
p_ridge = candidates['Ridge (alpha=300)'].predict(X_test)
p_svr = candidates['SVR (RBF, C=1.0)'].predict(X_test)
p_blend = 0.5 * p_ridge + 0.5 * p_svr
mae_b = mean_absolute_error(y_test, p_blend)
rmse_b = np.sqrt(mean_squared_error(y_test, p_blend))
r2_b = r2_score(y_test, p_blend)
corr_b, _ = stats.pearsonr(y_test, p_blend)
print(f"{'Blend (Ridge + SVR)':<25} {mae_b:>7.3f} {rmse_b:>7.3f} {r2_b:>7.4f} {corr_b:>10.4f}")

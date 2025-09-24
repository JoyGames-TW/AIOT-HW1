# 需要的套件
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import math

# ================
# 2 & 3. 資料理解 + 準備
# ================
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])  # 全為數值特徵
y = data.frame["target"]

num_features = X.columns.tolist()

# 數值前處理：缺失值（理論上此資料無缺失） + 標準化
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 若含類別特徵，可在此加入 OneHotEncoder，並用 ColumnTransformer 匯總
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    # 例：("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# 訓練 / 測試切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =================
# 4. 建模：基線與正則化模型
# =================

# 4.1 基線：Ordinary Least Squares
baseline_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LinearRegression())
])

baseline_pipe.fit(X_train, y_train)
y_pred_base = baseline_pipe.predict(X_test)

def report(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{label}] R2={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")

report(y_test, y_pred_base, "Baseline LinearRegression")

# 4.2 正則化：以 GridSearchCV 對 Ridge / Lasso / ElasticNet 調參
cv = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid_ridge = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
}

ridge_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", Ridge(random_state=42))
])

ridge_gs = GridSearchCV(ridge_pipe, param_grid=param_grid_ridge, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
ridge_gs.fit(X_train, y_train)

param_grid_lasso = {
    "model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0]
}
lasso_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", Lasso(random_state=42, max_iter=10000))
])
lasso_gs = GridSearchCV(lasso_pipe, param_grid=param_grid_lasso, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
lasso_gs.fit(X_train, y_train)

param_grid_en = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0],
    "model__l1_ratio": [0.2, 0.5, 0.8]
}
en_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", ElasticNet(random_state=42, max_iter=10000))
])
en_gs = GridSearchCV(en_pipe, param_grid=param_grid_en, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
en_gs.fit(X_train, y_train)

# 5. 評估：在測試集上評分
models = [
    ("Ridge(best)", ridge_gs.best_estimator_),
    ("Lasso(best)", lasso_gs.best_estimator_),
    ("ElasticNet(best)", en_gs.best_estimator_)
]
for name, est in models:
    pred = est.predict(X_test)
    report(y_test, pred, name)

# 5.1 交叉驗證（以最佳模型為例）
best_name, best_model = max(
    models,
    key=lambda kv: r2_score(y_test, kv[1].predict(X_test))
)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="r2", n_jobs=-1)
print(f"[{best_name}] CV R2 mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 5.2 殘差分析（數值）：觀察殘差分佈與均值
residuals = y_test - best_model.predict(X_test)
print(f"Residual mean={residuals.mean():.6f}, std={residuals.std():.6f}")

# =================
# 6. 部署：保存整個 Pipeline（前處理 + 模型）
# =================
joblib.dump(best_model, "linreg_pipeline.joblib")
print("Model saved to linreg_pipeline.joblib")

# （上線推論時）
# loaded = joblib.load("linreg_pipeline.joblib")
# y_new = loaded.predict(X_new_dataframe)  # X_new_dataframe 欄位需與訓練一致
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

from preprocess import load_data, preprocessing_pipeline, split_data

def eval_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(f"Model: {name}")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²:   {r2:.4f}\n")
    return r2

def train():
    df = load_data('../data/insurance.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = preprocessing_pipeline()
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    best_r2 = -1
    best_model_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        cur_r2 = eval_model(name, pipeline, X_test, y_test)
        if cur_r2 > best_r2:
            best_r2 = cur_r2
            best_model_pipeline = pipeline
    
    if best_model_pipeline:
        joblib.dump(best_model_pipeline, '../models/model.pkl')
        print(f"R^2 of best model: {best_r2:.4f}")

if __name__ == "__main__":
    train()
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Scoring
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score

# Algorithms
from sklearn.linear_model import Ridge
import xgboost as xgb 

# Custom import
from preprocess import *

def run(orig_df, df, target):
    target_name = target[0]
    pre_df, new_df, target_label = drop_empty_rows_for_target_label(orig_df, df, target_name)
    imputed_df = impute_data(new_df, method="knn", arg=5)
    
    vif = pd.DataFrame()
    vif["variables"] = imputed_df.columns
    vif["VIF"] = [variance_inflation_factor(imputed_df.values, i) for i in range(imputed_df.shape[1])]
    
    testsize = 0.4
    X_train, X_test, y_train, y_test = train_test_split(imputed_df, target_label, test_size=testsize, random_state=30)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    baseline_model = Ridge(random_state=30, **target[1])
    scores = cross_val_score(baseline_model, imputed_df, target_label, cv=10)
    baseline_cv = scores.mean()
    baseline_model.fit(X_train_scaled, y_train)
    
    predictions = baseline_model.predict(X_test_scaled)
    baseline_score = baseline_model.score(X_test_scaled, y_test)
    baseline_mse = mean_squared_error(y_test, predictions, squared=False)
    
    xgb_params = target[2]
    xg_reg = xgb.XGBRegressor(**xgb_params, random_state=30)
    scores = cross_val_score(xg_reg, imputed_df, target_label, cv=10)
    xgb_cv = scores.mean()
    xg_reg.fit(X_train,y_train)
    xgb_predicts = xg_reg.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_predicts)
    xgb_mse = mean_squared_error(y_test, xgb_predicts, squared=False)
    
    results = pd.DataFrame(data={
    'Algorithm': ['Baseline', 'XGBoost'],
    'R2': [baseline_score, xgb_r2 ],
    'MeanCV': [baseline_cv, xgb_cv ],
    'MSE': [baseline_mse,  xgb_mse ]
    })
    
    print(f'{target_name} - Baseline: R2 = {round(baseline_cv, 3)} RMSE = {round(baseline_mse, 3)}'
          f' | XGB: R2 = {round(xgb_cv, 3)} RMSE = {round(xgb_mse, 3)} ')
    return { target_name: (results, baseline_model, xg_reg, imputed_df, vif) }

def create_df_for_results(results):
    """Creates a DataFrame our of the dictionary holding the results
    """
    new_dataframe_collection = {}
    
    df = pd.DataFrame(columns=["Algorithm", "R2", "MeanCV", "MSE", "Target"])

    for result in results:
        target = pd.DataFrame(columns=["Target"], data=[result,result])
        full_row_df =  results[result][0].join(target)

        df = df.append(full_row_df,sort=False)
        
    return df

def get_results_from_regressors(targets_with_params, orig_df, df):
    results_dict = {}

    for target_with_params in targets_with_params:
        results_dict.update(run(orig_df, df, target_with_params))
        
    return results_dict, create_df_for_results(results_dict)

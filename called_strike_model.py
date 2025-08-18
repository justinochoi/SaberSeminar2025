
import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
import catboost as cb 
import optuna 
import joblib 

cards_data = pd.read_csv("data/cards_data.csv") 

# get only called strikes / balls / hbp 
cards_takes = cards_data[cards_data['description'].isin(['ball','called_strike','hit_by_pitch'])]  
cards_takes['description'].value_counts() 

# split into features / response var 
X = cards_takes[['plate_x','plate_z','sz_bot','sz_top','pitcher_hand','batter_hand']] 
X['pitcher_hand'] = pd.Categorical(X['pitcher_hand'])  
X['batter_hand'] = pd.Categorical(X['batter_hand'])  
y = cards_takes['description'] 

# set up optuna study 
def cb_objective(trial): 
    
    params = {
      'iterations': 500, 
      'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True), 
      "depth": trial.suggest_int("depth", 1, 10),
      "subsample": trial.suggest_float("subsample", 0.05, 1.0),
      "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
      "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=76)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y): 
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx] 
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = cb.CatBoostClassifier(**params, random_seed=76, 
                                      cat_features=['pitcher_hand','batter_hand'],
                                      bootstrap_type='Bernoulli') 
        model.fit(X_train, y_train, verbose=0)
        preds = model.predict_proba(X_val) 
        
        ll_score = log_loss(y_val, preds)
        cv_scores.append(ll_score) 
        
    return np.mean(cv_scores)
        
study = optuna.create_study(direction='minimize') 
study.optimize(cb_objective, n_trials=20)
best_params = study.best_params 

called_strike_mod = cb.CatBoostClassifier(
    iterations = 500, 
    learning_rate = best_params['learning_rate'], 
    depth = best_params['depth'], 
    subsample = best_params['subsample'], 
    colsample_bylevel = best_params['colsample_bylevel'], 
    min_data_in_leaf = best_params['min_data_in_leaf'], 
    random_seed=76, 
    cat_features=['pitcher_hand','batter_hand'], 
    bootstrap_type='Bernoulli'
    )

called_strike_mod.fit(X, y)
joblib.dump(called_strike_mod, "models/called_strike_mod.pkl") 


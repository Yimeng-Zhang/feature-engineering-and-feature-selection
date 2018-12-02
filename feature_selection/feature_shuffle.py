import pandas as pd
#import numpy as np


from sklearn.ensemble import RandomForestClassifier #, RandomForestRegressor
from sklearn.metrics import roc_auc_score #, mean_squared_error

# 2018.11.28 Created by Eamon.Zhang


def feature_shuffle_rf(X_train,y_train,max_depth=None,class_weight=None,top_n=15,n_estimators=50,random_state=0):
    
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, (model.predict_proba(X_train))[:, 1])
    feature_dict = {}

    # selection  logic
    for feature in X_train.columns:
        X_train_c = X_train.copy().reset_index(drop=True)
        y_train_c = y_train.copy().reset_index(drop=True)
        
        # shuffle individual feature
        X_train_c[feature] = X_train_c[feature].sample(frac=1,random_state=random_state).reset_index(
            drop=True)
        #print(X_train_c.isnull().sum())
        # make prediction with shuffled feature and calculate roc-auc
        shuff_auc = roc_auc_score(y_train_c,
                                  (model.predict_proba(X_train_c))[:, 1])
        #print(shuff_auc)
        # save the drop in roc-auc
        feature_dict[feature] = (train_auc - shuff_auc)
        #print(feature_dict)
    
    auc_drop = pd.Series(feature_dict).reset_index()
    auc_drop.columns = ['feature', 'auc_drop']
    auc_drop.sort_values(by=['auc_drop'], ascending=False, inplace=True)
    selected_features = auc_drop[auc_drop.auc_drop>0]['feature']

    return auc_drop, selected_features


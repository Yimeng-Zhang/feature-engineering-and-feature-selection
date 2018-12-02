#import pandas as pd
#import numpy as np

from sklearn.ensemble import RandomForestClassifier #, RandomForestRegressor
from sklearn.metrics import roc_auc_score #, mean_squared_error

# 2018.12.02 Created by Eamon.Zhang


def recursive_feature_elimination_rf(X_train,y_train,X_test,y_test,
                                     tol=0.001,max_depth=None,
                                     class_weight=None,
                                     top_n=15,n_estimators=50,random_state=0):
    
   
    features_to_remove = []
    count = 1
    # initial model using all the features
    model_all_features = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)
    model_all_features.fit(X_train, y_train)
    y_pred_test = model_all_features.predict_proba(X_test)[:, 1]
    auc_score_all = roc_auc_score(y_test, y_pred_test)
    
    for feature in X_train.columns:
        print()
        print('testing feature: ', feature, ' which is feature ', count,
          ' out of ', len(X_train.columns))
        count += 1
        model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)
        
        # fit model with all variables minus the removed features
        # and the feature to be evaluated
        model.fit(X_train.drop(features_to_remove + [feature], axis=1), y_train)
        y_pred_test = model.predict_proba(
                    X_test.drop(features_to_remove + [feature], axis=1))[:, 1]    
        auc_score_int = roc_auc_score(y_test, y_pred_test)
        print('New Test ROC AUC={}'.format((auc_score_int)))
    
        # print the original roc-auc with all the features
        print('All features Test ROC AUC={}'.format((auc_score_all)))
    
        # determine the drop in the roc-auc
        diff_auc = auc_score_all - auc_score_int
    
        # compare the drop in roc-auc with the tolerance
        if diff_auc >= tol:
            print('Drop in ROC AUC={}'.format(diff_auc))
            print('keep: ', feature)
            
        else:
            print('Drop in ROC AUC={}'.format(diff_auc))
            print('remove: ', feature)
            
            # if the drop in the roc is small and we remove the
            # feature, we need to set the new roc to the one based on
            # the remaining features
            auc_score_all = auc_score_int
            
            # and append the feature to remove to the list
            features_to_remove.append(feature)
    print('DONE!!')
    print('total features to remove: ', len(features_to_remove))  
    features_to_keep = [x for x in X_train.columns if x not in features_to_remove]
    print('total features to keep: ', len(features_to_keep))
    
    return features_to_keep


def recursive_feature_addition_rf(X_train,y_train,X_test,y_test,
                                     tol=0.001,max_depth=None,
                                     class_weight=None,
                                     top_n=15,n_estimators=50,random_state=0):
    
   
    features_to_keep = [X_train.columns[0]]
    count = 1
    # initial model using only one feature
    model_one_feature = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)
    model_one_feature.fit(X_train[[X_train.columns[0]]], y_train)
    y_pred_test = model_one_feature.predict_proba(X_test[[X_train.columns[0]]])[:, 1]  
    auc_score_all = roc_auc_score(y_test, y_pred_test)
    
    for feature in X_train.columns[1:]:
        print()
        print('testing feature: ', feature, ' which is feature ', count,
          ' out of ', len(X_train.columns))
        count += 1
        model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)
        
        # fit model with  the selected features
        # and the feature to be evaluated
        model.fit(X_train[features_to_keep + [feature]], y_train)
        y_pred_test = model.predict_proba(
                    X_test[features_to_keep + [feature]])[:, 1]    
        auc_score_int = roc_auc_score(y_test, y_pred_test)
        print('New Test ROC AUC={}'.format((auc_score_int)))
    
        # print the original roc-auc with all the features
        print('All features Test ROC AUC={}'.format((auc_score_all)))
    
        # determine the drop in the roc-auc
        diff_auc = auc_score_int - auc_score_all
    
        # compare the drop in roc-auc with the tolerance
        if diff_auc >= tol:
            # if the increase in the roc is bigger than the threshold
            # we keep the feature and re-adjust the roc-auc to the new value
            # considering the added feature
            print('Increase in ROC AUC={}'.format(diff_auc))
            print('keep: ', feature)
            auc_score_all = auc_score_int
            features_to_keep.append(feature)
        else:
            print('Increase in ROC AUC={}'.format(diff_auc))
            print('remove: ', feature)          

    print('DONE!!')
    print('total features to keep: ', len(features_to_keep))  
   
    return features_to_keep
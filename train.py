from sklearn import svm, linear_model
from xgboost import XGBRegressor
from Dataset import Dataset
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor

def train(if_test=False):
    #define a xgb model
    xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    silent=True,
    objective='reg:linear')
    #define a random forest model
    rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto')

    #train models
    train_path = '/home/sysadm/train_data'
    dataset = Dataset(blank_fill = 'avg', window_size = 7, use_KG = False, use_log = False, test_last = False, path=train_path)
    xgb.fit(dataset.X, dataset.Y)
    print("xgb model trained")

    if if_test:
        test_path = '/home/sysadm/test_data'
        test_data = Dataset(blank_fill = 'avg', window_size = 7, use_KG = False, use_log = False, test_last = False, path=test_path)

        pred_y = xgb.predict(test_data.X)
        #save the prediction result in json file
        with open('prediction.json', 'w') as f:
            json.dump(pred_y.tolist(), f)

    #find the most important features
    count = 0
    res = {}
    feature_names = dataset.get_colnames()
    for i in np.argsort(xgb.feature_importances_)[::-1]:
        if count < 10 and i<len(feature_names):
            #save the most important features in res
            res[feature_names[i]] = xgb.feature_importances_[i]
            count += 1
        if count == 10:
            break
    #save the most important features in json file
    with open('feature_importance.json', 'w') as f:
        json.dump(res, f)
    

if __name__ == "__main__":
    
    train(if_test=False)
        

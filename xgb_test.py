
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import random
from math import exp
import xgboost as xgb
import re
import matplotlib
#get_ipython().magic('matplotlib inline')
import datetime as dt
from functions import *
import operator
def InitSettings():
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.02
    param['max_depth'] = 5
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = 21
    param['nthread'] = 8
    param['lambda']=1.0
    param ['alpha']=0.0 #[default=0]


    maxstat={}
    maxstat["manager_id"]=30
    maxstat["building_id"]=30
    maxstat["street_address"]=30

    howtouseID={} # 0= H-M-L fractions 1 columns with 0-1 ,2 =nothing

    howtouseID['manager_id']=0
    howtouseID['building_id']=0
    howtouseID['street_address']=0





    columns_for_remove=["photos",
                   "description",
                   "interest_level", "created","manager_id",
                   "building_id","display_address", "street_address",#'time','listing_id',
                    'street_address_new', 'building_id_new',
                    'street_address_new_new','pred0_low','low','pred0_medium','medium','pred0_high','high',]
    others=dict()
    others["countvectorizer_max_features"]=50
    others["num_rounds"]=30
    others["withrest"]=True
    others["addlabelsasint"]=False
    others["clean_street_building_ids"]=0
    alllparams=dict()
    alllparams['xgb']=param
    alllparams['maxstat']=maxstat
    alllparams['howtouseID']=howtouseID
    alllparams['others']=others
    alllparams['columns_for_remove']=columns_for_remove
    return alllparams


def find_objects_with_only_one_record(df_train,df_test,feature_name):
    temp = pd.concat([df_train[feature_name].reset_index(),
                      df_test[feature_name].reset_index()])
    temp = temp.groupby(feature_name, as_index = False).count()
    return temp[temp['index'] == 1]

def categorical_average(df_train,df_test,variable, y, pred_0, feature_name):
    def calculate_average(sub1, sub2):
        s = pd.DataFrame(data = {
                                 variable: sub1.groupby(variable, as_index = False).count()[variable],
                                 'sumy': sub1.groupby(variable, as_index = False).sum()['y'],
                                 'avgY': sub1.groupby(variable, as_index = False).mean()['y'],
                                 'cnt': sub1.groupby(variable, as_index = False).count()['y']
                                 })

        tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable)
        del tmp['index']
        tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
        tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0
        lambda_val = None
        k=5.0
        f=1.0
        r_k=0.01
        g = 1.0
        def compute_beta(row):
            cnt = row['cnt'] if row['cnt'] < 200 else float('inf')
            return 1.0 / (g + exp((cnt - k) / f))

        if lambda_val is not None:
            tmp['beta'] = lambda_val
        else:
            tmp['beta'] = tmp.apply(compute_beta, axis = 1)

        tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred_0'],
                                   axis = 1)

        tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred_0']
        tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred_0']
        tmp['random'] = np.random.uniform(size = len(tmp))
        tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] *(1 + (row['random'] - 0.5) * r_k),
                                   axis = 1)

        return tmp['adj_avg'].ravel()

    #cv for training set
    k_fold = StratifiedKFold(5)
    print(feature_name)
    df_train[feature_name] = -999
    for (train_index, cv_index) in k_fold.split(np.zeros(len(df_train)),
                                                df_train['interest_level'].ravel()):
        sub = pd.DataFrame(data = {variable: df_train[variable],
                                   'y': df_train[y],
                                   'pred_0': df_train[pred_0]})

        sub1 = sub.iloc[train_index]
        sub2 = sub.iloc[cv_index]
        df_train.loc[cv_index, feature_name] = calculate_average(sub1, sub2)

    #for test set
    sub1 = pd.DataFrame(data = {variable: df_train[variable],
                                'y': df_train[y],
                                'pred_0': df_train[pred_0]})
    sub2 = pd.DataFrame(data = {variable: df_test[variable],
                                'y': df_test[y],
                                'pred_0': df_test[pred_0]})
    df_test.loc[:, feature_name] = calculate_average(sub1, sub2)


def transform_data(X,global_prob,feature_transform):
    #add features
    feat_sparse = feature_transform.transform(X["features"])
    vocabulary = feature_transform.vocabulary_
    del X['features']
    X1 = pd.DataFrame([ pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0]) ])
    X1.columns = list(sorted(vocabulary.keys()))
    X = pd.concat([X.reset_index(), X1.reset_index()], axis = 1)
    del X['index']

    X["num_photos"] = X["photos"].apply(len)
    X['created'] = pd.to_datetime(X["created"])
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X['price_per_bed'] = X['price'] / X['bedrooms']
    X['price_per_room'] = X['price'] / (X['bathrooms'] + X['bedrooms'] )

    X["created"] = pd.to_datetime(X["created"])
    X["created_year"] = X["created"].dt.year
    X["created_month"] = X["created"].dt.month
    X["created_day"] = X["created"].dt.day
    X['created_hour'] = X["created"].dt.hour
    X['created_weekday'] = X['created'].dt.weekday
    X['created_week'] = X['created'].dt.week
    X['created_quarter'] = X['created'].dt.quarter
    X['created_weekend'] = ((X['created_weekday'] == 5) & (X['created_weekday'] == 6))
    X['created_wd'] = ((X['created_weekday'] != 5) & (X['created_weekday'] != 6))
    X['created'] = X['created'].map(lambda x: float((x - dt.datetime(1899, 12, 30)).days) + (float((x - dt.datetime(1899, 12, 30)).seconds) / 86400))


    X['low'] = 0
    X.loc[X['interest_level'] == 0, 'low'] = 1
    X['medium'] = 0
    X.loc[X['interest_level'] == 1, 'medium'] = 1
    X['high'] = 0
    X.loc[X['interest_level'] == 2, 'high'] = 1


    X['pred0_low'] = global_prob[0]
    X['pred0_medium'] = global_prob[1]
    X['pred0_high'] = global_prob[2]


    return X


def remove_columns(X,columns):
    for c in columns:
        if c in X.columns:
            del X[c]




def LoadData(filename,settings):
    X_train =LoadTrain("train{}.json".format(filename))
    X_test = LoadTest("test{}.json".format(filename))

    feature_transform = CountVectorizer(stop_words='english', max_features=settings['others']['countvectorizer_max_features'])

    train_size = len(X_train)
    low_count = len(X_train[X_train['interest_level'] == 0])
    medium_count = len(X_train[X_train['interest_level'] == 1])
    high_count = len(X_train[X_train['interest_level'] == 2])

    global_prob=[low_count/train_size,medium_count/train_size,high_count/train_size]

    X_train['features'] = X_train["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
    X_test['features'] = X_test["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))


    feature_transform.fit(list(X_train['features']) + list(X_test['features']))
    print("Starting transformations")
    X_train = transform_data(X_train,global_prob,feature_transform)
    X_test = transform_data(X_test,global_prob,feature_transform)


    if settings['others']["clean_street_building_ids"]>0:
        print("Clean Street building 1")
        X_train=CreateSDN(X_train)
        trainBtoA,trainAtoB=GetDict(X_train)
        X_test=CreateSDN(X_test)
        testBtoA,testAtoB=GetDict(X_test)
        if settings['others']["clean_street_building_ids"]>1:
            print("Clean Street building 2")
            X_train['building_id_new']=X_train['building_id']
            X_train['street_address_new_new']=X_train['street_address_new']
            X_test['building_id_new']=X_test['building_id']
            X_test['street_address_new_new']=X_test['street_address_new']
            X_train=CleanIDstreet(X_train,trainBtoA,trainAtoB)
            X_train=CleanIDstreet(X_train,testBtoA,testAtoB)
            X_test=CleanIDstreet(X_test,trainBtoA,trainAtoB)
            X_test=CleanIDstreet(X_test,testBtoA,testAtoB)
            X_train['building_id']=X_train['building_id_new']
            X_test['building_id']=X_test['building_id_new']
            X_train['street_address']=X_train['street_address_new_new']
            X_test['street_address']=X_test['street_address_new_new']
        else:
            X_train['street_address']=X_train['street_address_new']
            X_test['street_address']=X_test['street_address_new']

    print("Normalizing high cordiality data...")
    for i in ['manager_id','street_address','building_id']:
        flag=settings['howtouseID'][i]
        if flag==1:
            AddColumns(X_train,labels,i)
            AddColumns(X_test,labels,i)
        elif flag==0:
            final,labels=GetFractionsforIDStreet(X_train,i,settings['maxstat'][i],settings['others']["withrest"])
            final=dict(zip(labels,final))
            AddColumnsLMHIDStreet(X_train,i,final)
            AddColumnsLMHIDStreet(X_test,i,final)
        elif flag==2:
            with_one_lot = find_objects_with_only_one_record(X_train,X_test,i)
            X_train.loc[X_train[i].isin(with_one_lot[i].ravel()),i] = "-1"
            X_test.loc[X_test[i].isin(with_one_lot[i].ravel()),i] = "-1"
            categorical_average(X_train,X_test,i, "low", "pred0_low", i + "_mean_low")
            categorical_average(X_train,X_test,i, "medium", "pred0_medium", i + "_mean_medium")
            categorical_average(X_train,X_test,i, "high", "pred0_high", i + "_mean_high")
        if settings['others']['addlabelsasint']:
            encoder = LabelEncoder()
            encoder.fit(list(X_train[i]) + list(X_test[i]))
            X_train["{}_label".format(i)] = encoder.transform(X_train[i].ravel())
            X_test["{}_label".format(i)] = encoder.transform(X_test[i].ravel())
    return X_train,X_test



def RunXGB(X_train,X_test,settings,filename,timestamp):
    logs=dict()
    y = X_train['interest_level'].ravel()
    remove_columns(X_train,settings['columns_for_remove'])
    xgtrain = xgb.DMatrix(X_train, label=y)
    results=True
    if filename!="":
        results=False
        ytest=X_test['interest_level'].ravel()
        remove_columns(X_test,settings['columns_for_remove'])
        xgtest = xgb.DMatrix(X_test, label=ytest)
        watchlist  = [ (xgtrain,'train'),(xgtest,'eval')]
    else:
        remove_columns(X_test,settings['columns_for_remove'])
        xgtest = xgb.DMatrix(X_test)
        watchlist  = [ (xgtrain,'train')]
    clf = xgb.train(settings['xgb'], xgtrain, settings['others']['num_rounds'],watchlist,evals_result=logs)


    ids=X_test['listing_id'].ravel()
    xgtest = xgb.DMatrix(X_test)
    preds = clf.predict(xgtest)
    sub = pd.DataFrame(data = {'listing_id': ids})
    sub['low'] = preds[:, 0]
    sub['medium'] = preds[:, 1]
    sub['high'] = preds[:, 2]
    sub.to_csv("./test/test{}timestamp{}.csv".format(filename,timestamp), index = False, header = True)

    plt.rcParams["figure.figsize"] = [40,40]
    xgb.plot_importance(clf)
    plt.savefig("./test/importance{}timestamp{}.png".format(filename,timestamp))

    file_imp  = open("./test/importance{}timestamp{}.txt".format(filename,timestamp), "w")
    imp=clf.get_fscore()
    sorted_x = sorted(imp.items(), key=operator.itemgetter(1), reverse=True)
    for i in sorted_x:
        file_imp.write("{}={}\n".format(i[0],i[1]))
    file_imp.close()

    if results:
        out=pd.DataFrame({'train':logs['train']['mlogloss']})
    else:
        out=pd.DataFrame({'train':logs['train']['mlogloss'],"test":logs['eval']['mlogloss']})
    out.to_csv("./test/logs{}timestamp{}.csv".format(filename,timestamp), index = False, header = True)




if __name__ == '__main__':
    if len(sys.argv) == 3:
        filename = sys.argv[1]
        if filename=='final':
            filename=""
        settingsfilename =sys.argv[2]
    else:
        quit()
    timestamp=dt.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    allparams=InitSettings()
    if settingsfilename!="default":
        allparams=ReadIn(settingsfilename,allparams)
    train,test=LoadData(filename,allparams)
    RunXGB(train,test,allparams,filename,timestamp)
    WriteSettings("./test/settings{}timestamp{}.txt".format(filename,timestamp),allparams,train.columns)

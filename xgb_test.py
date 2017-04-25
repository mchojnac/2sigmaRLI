
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
    others["cut_on_cleaning_feauters"]=-1.0
    others["cut_to_divide_on_building_id"]=-1
    others["cut_lan_log_selection"]=0.0002
    others['binsize']=-1.0
    others['addNNresults']=""
    others['diraddNNresults']=""
    alllparams=dict()
    alllparams['xgb']=param
    alllparams['maxstat']=maxstat
    alllparams['howtouseID']=howtouseID
    alllparams['others']=others
    alllparams['columns_for_remove']=columns_for_remove
    return alllparams


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

    ids=X_train['listing_id'].ravel()
    xgtrain = xgb.DMatrix(X_train)
    preds = clf.predict(xgtrain)
    sub = pd.DataFrame(data = {'listing_id': ids})
    sub['low'] = preds[:, 0]
    sub['medium'] = preds[:, 1]
    sub['high'] = preds[:, 2]
    sub.to_csv("./test/train{}timestamp{}.csv".format(filename,timestamp), index = False, header = True)


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
    test=RemoveUncommon(test,train)
    if allparams["others"]["cut_to_divide_on_building_id"]>0:
        train1,train2,test1,test2=DivideDF(train,test,"building_id",allparams["others"]["cut_to_divide_on_building_id"])
        filename1=filename+str(len(test1))
        filename2=filename+str(len(test2))
        RunXGB(train1,test1,allparams,filename1+"part1",timestamp)
        WriteSettings("./test/settings{}part1timestamp{}.txt".format(filename1,timestamp),allparams,train1.columns)
        columnswithbid=list()
        for i in train1.columns:
            if i.find("building_id")>-1:
                columnswithbid.append(i)
        remove_columns(train2,columnswithbid)
        remove_columns(test2,columnswithbid)
        RunXGB(train2,test2,allparams,filename2+"part2",timestamp)
        WriteSettings("./test/settings{}part2timestamp{}.txt".format(filename2,timestamp),allparams,train2.columns)
    else:
        RunXGB(train,test,allparams,filename,timestamp)
        WriteSettings("./test/settings{}timestamp{}.txt".format(filename,timestamp),allparams,train.columns)

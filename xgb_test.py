
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
import datetime
from functions import *

def transform_data(X):
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

    return X



def remove_columns(X,columns):
    for c in columns:
        if c in X.columns:
            del X[c]

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
#param['lambda']=1.5
#param ['alpha']=0.4 #[default=0]
num_rounds = 3500

maxstat={}
maxstat["manager_id"]=30
maxstat["building_id_new"]=30
maxstat["street_address_new_new"]=30

howtouseID={} # 0= H-M-L fractions 1 columns with 0-1 ,2 =nothing

howtouseID['manager_id']=1
howtouseID['building_id_new']=1
howtouseID['street_address_new_new']=1

withrest=False

filename="fract0_15rs0"

columns_for_remove=["photos",
               "description",
               "interest_level", "created","manager_id",
               "building_id","display_address", "street_address",#'time','listing_id',
                'street_address_new', 'building_id_new',
                'street_address_new_new']
timestamp=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
countvectorizer_max_features=50


X_train =LoadTrain("train{}.json".format(filename))
X_test = LoadTest("test{}.json".format(filename))


feature_transform = CountVectorizer(stop_words='english', max_features=countvectorizer_max_features)


X_train['features'] = X_train["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
X_test['features'] = X_test["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))


feature_transform.fit(list(X_train['features']) + list(X_test['features']))


print("Starting transformations")
X_train = transform_data(X_train)
X_test = transform_data(X_test)
y = X_train['interest_level'].ravel()

X_train=CreateSDN(X_train)
trainBtoA,trainAtoB=GetDict(X_train)
print("step1")
X_test=CreateSDN(X_test)
testBtoA,testAtoB=GetDict(X_test)
print("step2")
X_train['building_id_new']=X_train['building_id']
X_train['street_address_new_new']=X_train['street_address_new']
X_test['building_id_new']=X_test['building_id']
X_test['street_address_new_new']=X_test['street_address_new']
X_train=CleanIDstreet(X_train,trainBtoA,trainAtoB)
X_train=CleanIDstreet(X_train,testBtoA,testAtoB)
print("step3")
X_test=CleanIDstreet(X_test,trainBtoA,trainAtoB)
X_test=CleanIDstreet(X_test,testBtoA,testAtoB)
print("step4")

print("Normalizing high cordiality data...")
for i in ['manager_id','street_address_new_new','building_id_new']:
    final,labels=GetFractionsforIDStreet(X_train,i,maxstat[i],withrest)
    if howtouseID[i]==1:
        AddColumns(X_train,labels,i)
        AddColumns(X_test,labels,i)
    elif howtouseID[i]==0:
        final=dict(zip(labels,final))
        AddColumnsLMHIDStreet(X_train,i,final)
        AddColumnsLMHIDStreet(X_test,i,final)
    else:
        continue



ids=X_test['listing_id'].ravel()

logs=dict()
remove_columns(X_train,columns_for_remove)
xgtrain = xgb.DMatrix(X_train, label=y)
results=True
if 'interest_level' in X_test.columns:
    results=False
    ytest=X_test['interest_level'].ravel()
    remove_columns(X_test,columns_for_remove)
    xgtest = xgb.DMatrix(X_test, label=ytest)
    watchlist  = [ (xgtrain,'train'),(xgtest,'eval')]
else:
    remove_columns(X_test,columns_for_remove)
    xgtest = xgb.DMatrix(X_test)
    watchlist  = [ (xgtrain,'train')]
clf = xgb.train(param, xgtrain, num_rounds,watchlist,evals_result=logs)



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

plt.rcParams["figure.figsize"] = [7,7]
plt.plot(logs['train']['mlogloss'],'g--')
if results==False:
    plt.plot(logs['eval']['mlogloss'],'r--')

if results:
    out=pd.DataFrame({'train':logs['train']['mlogloss']})
else:
    out=pd.DataFrame({'train':logs['train']['mlogloss'],"test":logs['eval']['mlogloss']})
out.to_csv("./test/logs{}timestamp{}.csv".format(filename,timestamp), index = False, header = True)

file_object  = open("./test/settings{}timestamp{}.txt".format(filename,timestamp), "w")

file_object.write("---------------XBOOST Part------------------\n")
for i in sorted(param.keys()):
    file_object.write("par {} = {} \n".format(i,param[i]))


file_object.write("---------------Removed Columns------------------\n")
for i in columns_for_remove:
    file_object.write("{}\n".format(i))



file_object.write("---------------Others------------------\n")
for i in sorted(maxstat.keys()):
    file_object.write("par n cut {} = {} \n".format(i,maxstat[i]))
for i in sorted(howtouseID.keys()):
    file_object.write("par what {} = {} \n".format(i,howtouseID[i]))
file_object.write("num_rounds= {} \n".format(num_rounds))
file_object.write("withrest= {} \n".format(withrest))
file_object.write("countvectorizer_ max_features= {} \n".format(countvectorizer_max_features))
file_object.write("Ncolumns {} {} \n".format(len(X_train.columns),len(X_test.columns)))

file_object.close()

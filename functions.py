import operator
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import math
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import datetime as dt
from math import exp

def AddFeatures(df,listoffeatures):
    tabel=np.zeros((len(df),len(listoffeatures)),dtype=int)
    for i_id,i in  enumerate(train_df['features'].values):
        for j_id,j in enumerate(listoffeatures):
            if ", ".join(i).find(j)>=0:
                tabel[i_id][j_id]=1
    df_test=pd.DataFrame(tabel,columns=listoffeatures,index=df.index)
    return pd.concat([df,df_test], axis=1)

def Filter(df,listoffeatures,clean=True):
    df['lenf']=len(df['features'])
    df['ifphoto']=len(df['photos'])
    df['lendescription']=len(df['description'])
    for i in range(len(df)):
        df['lenf'].values[i]=len(df['features'].values[i])
        df['ifphoto'].values[i]=len(df['photos'].values[i])
        df['lendescription'].values[i] = len(df['description'].values[i])
    if len(listoffeatures)>0:
        df,listoffeatures=GetFeatures(df,listoffeatures)
        df['add'] = df[listoffeatures].sum(axis=1)
    if clean:
        #df=df.loc[df['price']>2500]
        df=df.loc[df['bedrooms']<=4]
        df=df.loc[df['bathrooms']>0]
        df=df.loc[df['bathrooms']<=2]
        df=df.loc[df['ifphoto']>0]
    return df

def GetListofFeatures(df):
    allf=dict()
    for i in df['features'].values:
        for j in i:
            if j in allf.keys():
                allf[j]+=1
            else:
                allf[j]=1
    sorted_allf = sorted(allf.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_allf,allf

def GetStreetMangerBulding (df,columnname):
    alls=dict()
    for i in df[columnname].values:
        if i in alls.keys():
            alls[i]+=1
        else:
            alls[i]=1
    sorted_alls = sorted(alls.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_alls,alls

def PlotOneFeatureTrain(train_df,title,limits):
    df_h=train_df.loc[train_df['interest_level']==2]
    df_m=train_df.loc[train_df['interest_level']==1]
    df_l=train_df.loc[train_df['interest_level']==0]
    #print(limits[0],limits[1],limits[2])
    pr1=plt.hist(df_l[title],bins=np.linspace(limits[0],limits[1],limits[2]),color="b",alpha=0.3)
    #print(limits[0],limits[1],limits[2])
    pr2=plt.hist(df_m[title],bins=np.linspace(limits[0],limits[1],limits[2]),color="g",alpha=0.3)
    pr3=plt.hist(df_h[title],bins=np.linspace(limits[0],limits[1],limits[2]),color="r",alpha=0.3)
    bins=(pr1[1][1:]+pr1[1][:-1])*0.5
    plt.show()
    fig, ax = plt.subplots()
    error1=[math.sqrt(pr1[0][i]*(pr2[0][i]+pr3[0][i]))/((pr1[0][i]+pr2[0][i]+pr3[0][i])*math.sqrt(pr1[0][i]+pr2[0][i]+pr3[0][i])) for i in  range(len(pr1[0]))]
    error2=[math.sqrt(pr2[0][i]*(pr1[0][i]+pr3[0][i]))/((pr1[0][i]+pr2[0][i]+pr3[0][i])*math.sqrt(pr1[0][i]+pr2[0][i]+pr3[0][i])) for i in  range(len(pr1[0]))]
    error3=[math.sqrt(pr3[0][i]*(pr2[0][i]+pr1[0][i]))/((pr1[0][i]+pr2[0][i]+pr3[0][i])*math.sqrt(pr1[0][i]+pr2[0][i]+pr3[0][i])) for i in  range(len(pr1[0]))]
    nor=sum(pr1[0])+sum(pr2[0])+sum(pr3[0])

    based1=[sum(pr1[0])/nor for i in  range(len(pr1[0]))]
    based2=[sum(pr2[0])/nor for i in  range(len(pr1[0]))]
    based3=[sum(pr3[0])/nor for i in  range(len(pr1[0]))]

    ax.errorbar(x=bins,y=pr1[0]/(pr1[0]+pr2[0]+pr3[0]),yerr=error1,color="b",marker='o')
    ax.errorbar(x=bins,y=pr2[0]/(pr1[0]+pr2[0]+pr3[0]),yerr=error2,color="r",marker='o')
    ax.errorbar(x=bins,y=pr3[0]/(pr1[0]+pr2[0]+pr3[0]),yerr=error3,color="g",linestyle='dotted',marker='o')
    ax.errorbar(x=bins,y=based1,color="b",linestyle='dotted')
    ax.errorbar(x=bins,y=based2,color="r",linestyle='dotted')
    ax.errorbar(x=bins,y=based3,color="g",linestyle='dotted')
    ax.set_title(title)
    plt.show()
    nor=sum(pr1[0])+sum(pr2[0])+sum(pr3[0])
    plt.plot(bins,(pr1[0]+pr2[0]+pr3[0])/nor,"-o")
    plt.show()
    return

def GetFeautresStats(train_df,fract=0.01):
    v=list()
    labels=list()
    sorted_allf,dictall = GetListofFeatures(train_df)
    nor=len(train_df)
    for iv in sorted_allf:
        if iv[1]/nor<fract:
            break
        labels.append(iv[0])
        train_df_f=train_df[[iv[0] in x for x in train_df['features']]]
        norf=len(train_df_f)
        norf1=len(train_df_f.loc[train_df_f['interest_level']==2])
        norf2=len(train_df_f.loc[train_df_f['interest_level']==1])
        norf3=len(train_df_f.loc[train_df_f['interest_level']==0])

        #print("{} {} {}".format(norf1/norf,norf2/norf,norf3/norf))
        v.append([norf1/norf,norf2/norf,norf3/norf,norf])
    return labels,v



def PlotFeautresfromTrain(train_df):
    v1=list()
    v2=list()
    v3=list()
    v4=list()
    based1=list()
    based2=list()
    based3=list()
    error1=list()
    error2=list()
    error3=list()
    labels,v = GetFeautresStats(train_df)
    nor=len(train_df)
    for iv in range(len(v)):

        norf1=v[iv][0]
        norf2=v[iv][1]
        norf3=v[iv][2]
        norf=v[iv][3]
        v1.append(norf1)
        v2.append(norf2)
        v3.append(norf3)
        v4.append(norf)

        based1.append(0.07778813421948452)
        based2.append(0.22752877289674178)
        based3.append(0.6946830928837737)

        error1.append(math.sqrt(norf1/norf*(1.0-norf1/norf))/math.sqrt(norf))
        error2.append(math.sqrt(norf2/norf*(1.0-norf2/norf))/math.sqrt(norf))
        error3.append(math.sqrt(norf3/norf*(1.0-norf3/norf))/math.sqrt(norf))
    fig, ax = plt.subplots()
    ax.errorbar(x=range(len(v1)),y=v1,yerr=error1,color="g",marker='o',linestyle='None')
    ax.errorbar(x=range(len(v2)),y=v2,yerr=error2,color="r",marker='o',linestyle='None')
    ax.errorbar(x=range(len(v3)),y=v3,yerr=error3,color="b",marker='o',linestyle='None')
    ax.errorbar(x=range(len(based1)),y=based1,color="g",linestyle='dotted')
    ax.errorbar(x=range(len(based2)),y=based2,color="r",linestyle='dotted')
    ax.errorbar(x=range(len(based3)),y=based3,color="b",linestyle='dotted')
    #ax.set_title(columnname)
    plt.xticks(range(len(v1)),labels,rotation='vertical')
    plt.show()
    plt.plot(v4)
    plt.show()
    return

def GetFractionsforIDStreet(train_df,columnname,cut=120,userest=True):
    labels=list()
    sorted_alls,alls = GetStreetMangerBulding (train_df,columnname)
    final=list()
    rest=train_df
    print(columnname)
    for iv in sorted_alls:
        if iv[1]<cut:
            break
        labels.append(iv[0])
        #print("{}".format(len(rest)))
        small=train_df.loc[train_df[columnname]==iv[0]]
        rest=rest.loc[rest[columnname]!=iv[0]]
        frac=[1.0/iv[1],1.0/iv[1],1.0/iv[1],iv[1]]
        for index,freq in small.groupby('interest_level'):
            frac[index]=len(freq)*frac[index]
        final.append(frac)
    labels.append('rest')
    lrest=len(rest)
    if lrest==0:
        lrest=1
    frac=[1/lrest,1/lrest,1/lrest,lrest]
    if userest:
        for index,freq in rest.groupby('interest_level'):
                frac[index]=len(freq)*frac[index]
    else:
        frac=[0.0,0.0,0.0,lrest]
    final.append(frac)
    final= np.array(final)
    return final,labels

def GetStatofSMIinTrain (columnname,train_df,cut=120):
  #  labels=list()
  #  sorted_alls,alls = GetStreetMangerBulding (train_df,columnname)
  #  final=list()
  #  for iv in sorted_alls:
  #      if iv[1]<cut:
  #          break
  #      labels.append(iv[0])
        #print("{}".format(iv))
  #      small=train_df.loc[train_df[columnname]==iv[0]]
  #      frac=[1.0/iv[1],1.0/iv[1],1.0/iv[1],iv[1]]
  #      for index,freq in small.groupby('interest_level'):
  #          frac[index]=len(freq)*frac[index]
  #      final.append(frac)
  #  final= np.array(final)
    #plt.subplot(211)
    final,labels=GetFractionsforIDStreet(train_df,columnname,cut)

    fig, ax = plt.subplots()

    nor=len(train_df)
    nor1=len(train_df.loc[train_df['interest_level']==2])
    nor2=len(train_df.loc[train_df['interest_level']==1])
    nor3=len(train_df.loc[train_df['interest_level']==0])

    based1=[nor3/nor for i in  range(len(final[:,0]))]
    based2=[nor2/nor for i in  range(len(final[:,0]))]
    based3=[nor1/nor for i in  range(len(final[:,0]))]

    error1=[math.sqrt(final[i,0]*(1.0-final[i,0]))/math.sqrt(final[i,3]) for i in  range(len(final[:,0]))]
    error2=[math.sqrt(final[i,1]*(1.0-final[i,1]))/math.sqrt(final[i,3]) for i in  range(len(final[:,1]))]
    error3=[math.sqrt(final[i,2]*(1.0-final[i,2]))/math.sqrt(final[i,3]) for i in  range(len(final[:,2]))]
#error1=[math.sqrt(pr1[i]*(pr2[i]+pr3[i])/((pr1[i]+pr2[i]+pr3[i])*math.sqrt(pr1[i]+pr2[i]+pr3[i]))) for i in  range(len(pr1[0]))]
#plt.plot(pr1[0]/(pr1[0]+pr2[0]+pr3[0]),"b--")
    ax.errorbar(x=range(len(final[:,0])),y=final[:,0],yerr=error1,color="b",marker='o',linestyle='None')
    ax.errorbar(x=range(len(final[:,1])),y=final[:,1],yerr=error2,color="r",marker='o',linestyle='None')
    ax.errorbar(x=range(len(final[:,2])),y=final[:,2],yerr=error3,color="g",marker='o',linestyle='None')
    ax.errorbar(x=range(len(final[:,0])),y=based1,color="b",linestyle='dotted')
    ax.errorbar(x=range(len(final[:,1])),y=based2,color="r",linestyle='dotted')
    ax.errorbar(x=range(len(final[:,2])),y=based3,color="g",linestyle='dotted')
    ax.set_title(columnname)
    plt.xticks(range(len(final[:,2])),labels,rotation='vertical')
    plt.show()
    fig2, ax2 = plt.subplots()
    plt.plot(final[:,3]/nor)
    ax2.set_yscale('log')
    plt.show()

   #plt.plot(final[:,0],'--g')
   # plt.plot(final[:,1],'--r')
   # plt.plot(final[:,2],'--b')
    return final

def LoadTrain(name="./train.json"):
    train_df = pd.read_json(name)
    train_df.head()
    train_df.reset_index(inplace=True)
    del train_df["index"]
    maping_intrest={'low':0,'medium':1,'high':2}
    train_df['interest_level'] = train_df['interest_level'].map(maping_intrest)
    train_df['features'] = train_df["features"].apply(lambda x: ["_".join(i.lower().split(" ")) for i in x])
    train_df['time']=train_df['created']
    start=pd.Timestamp('2016-04-01 22:23:31')
    train_df['time'] = train_df["time"].apply(lambda x:np.datetime64(x))
    train_df['time'] = train_df["time"].apply(lambda x:(x-start))
    train_df['time'] = train_df["time"].astype(int)
    return train_df

def LoadTest(name="./test.json"):
    df_test= pd.read_json(name)
    df_test.reset_index(inplace=True)
    del df_test["index"]
    if "interest_level" in df_test.columns:
        maping_intrest={'low':0,'medium':1,'high':2}
        df_test['interest_level'] =df_test['interest_level'].map(maping_intrest)
    else:
        df_test['interest_level'] = -1
    df_test['features'] = df_test["features"].apply(lambda x: ["_".join(i.lower().split(" ")) for i in x])
    df_test['time']=df_test['created']#.replace(" ","").replace("-","").replace(":","")[3:])
    start=pd.Timestamp('2016-04-01 22:23:31')
    df_test['time'] = df_test["time"].apply(lambda x:np.datetime64(x))
    df_test['time'] = df_test["time"].apply(lambda x:(x-start))
    df_test['time'] = df_test["time"].astype(int)
    return df_test

def Plot2Dict(train_dict,test_dict,train_l,test_l,zoom=False):
    train_f=list()
    test_f=list()
    for i in train_dict.keys():
        if i in test_dict.keys():
            train_f.append(train_dict[i]/train_l)
            test_f.append(test_dict[i]/test_l)
    plt.plot(train_f,test_f,"go")
    if zoom:
        plt.axis([0,0.02,0.0,0.02])

def CompareTrainTest(train,test,title,limits):
    hist_train=plt.hist(train[title],bins=np.linspace(limits[0],limits[1],limits[2]),color="b",alpha=0.3,normed=True)
    hist_test=plt.hist(test[title],bins=np.linspace(limits[0],limits[1],limits[2]),color="g",alpha=0.3,normed=True)
    plt.show()
    bins=(hist_train[1][1:]+hist_train[1][:-1])*0.5
    results=list()
    for i1,i2 in zip(hist_train[0],hist_test[0]):
        if (i1>0.0 and i2>0.0):
            results.append(i1/i2)
        else:
            results.append(0.0)
    plt.plot(bins,results)

def Std(df1,df2,columns):
    for i in columns:
        df1,df2=Cleanoutlayers(df1,df2,i)
        tmp=pd.concat([df1[i],df2[i]])
        mean=tmp.mean()
        std=tmp.std()+1e-8
        df1[i]=(df1[i]-mean)/std
        df2[i]=(df2[i]-mean)/std
    return

def AddColumnsLMHIDStreet(df_train,column,values) :
    df_train["{}_L".format(column)]=0.0
    df_train["{}_M".format(column)]=0.0
    df_train["{}_H".format(column)]=0.0
    for i in range(len(df_train)):
        if df_train[column].values[i] in values.keys():
            df_train["{}_L".format(column)].values[i]=values[df_train[column].values[i]][0]
            df_train["{}_M".format(column)].values[i]=values[df_train[column].values[i]][1]
            df_train["{}_H".format(column)].values[i]=values[df_train[column].values[i]][2]
        else:
            df_train["{}_L".format(column)].values[i]=values['rest'][0]
            df_train["{}_M".format(column)].values[i]=values['rest'][1]
            df_train["{}_H".format(column)].values[i]=values['rest'][2]
    return  df_train

def AddFeatureColumns(df_train,labels,values) :
    df_train["features_L"]=0.0
    df_train["features_M"]=0.0
    df_train["features_H"]=0.0
    for i in range(len(df_train)):
        if len(df_train['features'].values[i])==0:
            continue
        tmp=0
        h=0
        m=0.0
        l=0.0
        for ii,il in enumerate(labels):
            if il in df_train['features'].values[i]:
                h=h+values[ii][0]
                m=m+values[ii][1]
                l=l+values[ii][2]
                tmp=tmp+1.0
        if tmp>0.0:
            df_train["features_L"].values[i]=l/tmp
            df_train["features_M"].values[i]=m/tmp
            df_train["features_H"].values[i]=h/tmp
    return  df_train

def GetFeatures(df,listoffeatures):
    tabel=np.zeros((len(df),len(listoffeatures)),dtype=int)
    for i_id,i in  enumerate(df['features'].values):
        for j_id,j in enumerate(listoffeatures):
            if ", ".join(i).find(j)>=0:
                tabel[i_id][j_id]=1
    listoffeatures2=list()
    for i in listoffeatures:
        listoffeatures2.append(i.replace(" ",""))
    df_test=pd.DataFrame(tabel,columns=listoffeatures2,index=df.index)
    #return  pd.concat([df,df_test], axis=1)
    return pd.concat([df,df_test], axis=1),listoffeatures2

def testprediction(Y,Xprob):
    testv=0
    if len(Y)==0 or len(Xprob)==0 or len(Y)!=len(Xprob):
        return 100
    for i in range(len(Y)):
        testv+=math.log2(Xprob[i,np.argmax(Y[i])])
    return testv/len(Xprob)

def CreateSDN(df_train):
    df_train['street_address_new'] = df_train["street_address"].apply(lambda x: ' '.join(x.split()).lower())
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("\'",""))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st marks',"saint marks"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st. marks',"saint marks"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st nicholas',"saint nicholas"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st. nicholas',"saint nicholas"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st johns',"saint johns"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st. johns',"saint johns"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st pauls',"saint pauls"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('st. pauls',"saint pauls"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('stratford','satratford'))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('stockholm','sztockholm'))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('stuyvesan','situyvesan'))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('stanhope','sktanhope'))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('sterling','sgterling'))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('26th street 100 west',"100 west 26th street"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace('stanton','sftanton'))
    for ii,i in enumerate(df_train['street_address_new'].values):
        change=False
        for j in [" st"," st.", "street"]:
            pos=i.rfind(j)
            if pos>0:
                df_train['street_address_new'].values[ii]=i[:pos+len(j)]
                change=True
                break
        if change==False:
            df_train['street_address_new'].values[ii]=i+" st"

    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("street","st"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(".",""))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(",",""))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("-",""))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" east "," e "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" west "," w "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" nord "," n "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" south "," s "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("1st","1"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("2nd","2"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("3rd","3"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("4th","4"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("5th","5"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("6th","6"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("7th","7"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("8th","8"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("9th","9"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("0th","0"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("11th","11"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("12th","12"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace("13th","13"))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" pl "," "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" place "," "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" avenue "," "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" ave "," "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" blvd "," "))
    df_train['street_address_new'] = df_train["street_address_new"].apply(lambda x: x.replace(" boulevard "," "))
    return df_train

def GetDict(train,test):
    building_id_to_street=dict()
    address_to_building_id=dict()
    testb=set(test["building_id"].unique())
    trainb=set(train["building_id"].unique())
    allbid=testb.union(trainb)
    allbid.remove("0")
    n1=0
    for i in allbid:
        tmptest=test.loc[test["building_id"]==i]
        tmptrain=train.loc[train["building_id"]==i]
        #print("{} {}".format(len(tmp),len(tmp['street_address_new'].unique())))
        santest=set(tmptest['street_address_new'].unique())
        santrain=set(tmptrain['street_address_new'].unique())
        tmp=santest.union(santrain)
        if " st" in tmp:
            tmp.remove(" st")
        tmplen=[len(j) for j in tmp]
        tmpstr=[j for j in tmp]
        value=tmpstr[tmplen.index(min(tmplen))]
        m=value.find(" ")
        if value[:m].isnumeric()==False:
            m=-1
        for j in tmp:
            address_to_building_id[j]=i
        building_id_to_street[i]=value[m+1:]
        for j in test.index[test["building_id"]==i]:
            test['street_address'].values[j]=value[m+1:]
            test['street_address_new'].values[j]=value
        for j in train.index[train["building_id"]==i]:
            train['street_address'].values[j]=value[m+1:]
            train['street_address_new'].values[j]=value
    return building_id_to_street,address_to_building_id,train,test

def GetDict(train,test):
    building_id_to_street=dict()
    address_to_building_id=dict()
    dict_weights=dict()
    testb=set(test["building_id"].unique())
    trainb=set(train["building_id"].unique())
    allbid=testb.union(trainb)
    allbid.remove("0")
    for i in allbid:
        tmptest=test.loc[test["building_id"]==i]
        tmptrain=train.loc[train["building_id"]==i]
        santest=set(tmptest['street_address_new'].unique())
        santrain=set(tmptrain['street_address_new'].unique())
        tmp=santest.union(santrain)
        tmplen=[len(tmptest.loc[tmptest["street_address_new"]==j])+len(tmptrain.loc[tmptrain["street_address_new"]==j]) for j in tmp]
        tmpstr=[j for j in tmp]
        weight=max(tmplen)
        value=tmpstr[tmplen.index(weight)]
        m=value.find(" ")
        if value[:m].isnumeric()==False:
            m=-1
        if  weight>100:
            if value in address_to_building_id.keys():
                if weight>dict_weights[value]:
                    address_to_building_id[value]=i
                    dict_weights[value]=weight
            else:
                address_to_building_id[value]=i
                dict_weights[value]=weight
        building_id_to_street[i]=value[m+1:]
        for j in test.index[test["building_id"]==i]:
            test['street_address'].values[j]=value[m+1:]
            test['street_address_new'].values[j]=value
        for j in train.index[train["building_id"]==i]:
            train['street_address'].values[j]=value[m+1:]
            train['street_address_new'].values[j]=value
    return building_id_to_street,address_to_building_id,train,test


def CleanStreet(df,building_id_to_street):
    for i in range(len(df)):
        if df['building_id_new'].values[i]!='0':
            if df['building_id_new'].values[i] in building_id_to_street.keys():
                df['street_address_new_new'].values[i]=building_id_to_street[df['building_id_new'].values[i]]
        else:
            value=df['street_address_new'].values[i]
            m=value.find(" ")
            if value[:m].isnumeric()==False:
                m=-1
            df['street_address_new_new'].values[i]=value[m+1:]

    return  df
def CleanBuildingID(df,address_to_building_id):
    for i in range(len(df)):
        if df['street_address_new'].values[i] in address_to_building_id.keys():
            df['building_id_new'].values[i]=address_to_building_id[df['street_address_new'].values[i]]
    return  df

def FillMissingID(train,test):
    tmptest=test.loc[test["building_id_new"]=='0']
    tmptrain=train.loc[train["building_id_new"]=='0']
    santest=set(tmptest['street_address_new'].unique())
    santrain=set(tmptrain['street_address_new'].unique())
    tmp=santest.union(santrain)
    for (index,i) in enumerate(tmp):
        for j in test.index[(test["building_id_new"]=='0')&(test["street_address_new"]==i)]:
            test["building_id_new"].values[j]="Id_{}".format(index)
        for j in train.index[(train["building_id_new"]=='0') & (train["street_address_new"]==i)]:
            train["building_id_new"].values[j]="Id_{}".format(index)
    return train,test

def AddColumns(train,labels,column):
    for il,label in enumerate(labels):
        if len(label)>1: # chek not zero
            train["{}_{}".format(column,il)]= train[column].apply(lambda x: 1 if x==label else 0)


def ReadIn(filename,allparams):
    f=open(filename, 'r')
    for line in f.readlines():
        line=line.replace("\n","")
        if line[0]=="#":
            continue
        elif line[0]=="$":
            words=line.split(" ")
            if (words[1] in allparams.keys()):
                allparams[words[1]]=words[2:]
        else:
            words=line.split(" ")
            if len(words)==2:
                names=words[0].split("%")
                if len(names)==2:
                    if (names[0] in allparams.keys()) and (names[1] in allparams[names[0]].keys()):
                        if type(allparams[names[0]][names[1]])==str:
                            allparams[names[0]][names[1]]=str(words[1])
                        if type(allparams[names[0]][names[1]])==float:
                            allparams[names[0]][names[1]]=float(words[1])
                        if type(allparams[names[0]][names[1]])==int:
                            allparams[names[0]][names[1]]=int(words[1])
                        if type(allparams[names[0]][names[1]])==bool:
                            if words[1]=='True':
                                allparams[names[0]][names[1]]=True
                            else:
                                allparams[names[0]][names[1]]=False
    return allparams

def WriteSettings(filename,allparams,columns=None):
    file_object  = open(filename, "w")
    for i in sorted(allparams.keys()):
        if i=='columns_for_remove':
            file_object.write("$ columns_for_remove")
            for j in allparams[i]:
                file_object.write(" {}".format(j))
            file_object.write("\n")
        else:
            for j in sorted(allparams[i].keys()):
                file_object.write("{}%{} {}\n".format(i,j,allparams[i][j]))
    if len(columns)>0:
        file_object.write("#")
        for i in columns:
            file_object.write(" {}".format(i))
        file_object.write("\n")
    file_object.close()

def mergecolumns(df,tomerge,name):
    update=list()
    for i in tomerge:
        if i in df.columns:
            update.append(i)
    if len(update)==0:
        return df
    df[name]=df[update].sum(axis=1)
    for i in update:
        del df[i]
    return df

def RemoveLowSatColumns(df,cut=0.01):
    l=len(df)
    if l<1:
        return df
    for i in df.columns:
        if df[i].sum(0)/l<cut:
            del df[i]
    return df

def RemoveUncommon(df1,df2):
    for i in df1.columns:
        if (i in df2.columns)==False:
            del df1[i]
    return df1

def DivideDF(train,test,column,cut):
    testb=set(test[column].unique())
    trainb=set(train[column].unique())
    allbid=testb.union(trainb)
    allbid.remove('0')
    part1=list()
    part2=list()
    part2.append("0")
    for i in allbid:
        if len(train[train[column]==i])>cut:
             part1.append(i)
        else:
             part2.append(i)
    train1=train[train["building_id"].isin(part1)]
    train2=train[train["building_id"].isin(part2)]
    test1=test[test["building_id"].isin(part1)]
    test2=test[test["building_id"].isin(part2)]
    return train1,train2,test1,test2

def NewWay(train,test,cut=0.0002):
    df_all=pd.concat([train[["building_id",'latitude','longitude']],test[["building_id",'latitude','longitude']]])
    df_all=df_all.drop(df_all.index[df_all['building_id']=="0"])
    means=df_all.groupby("building_id", as_index = False).mean()
    train['building_id_new']="0"
    test['building_id_new']="0"
    for i in range(len(means)):
        buid=means['building_id'].values[i]
        la=means['latitude'].values[i]
        lo=means['longitude'].values[i]
        tmptest=test[test['building_id']==buid][['latitude','longitude']]
        tmptrain=train[train['building_id']==buid][['latitude','longitude']]
        tmp=pd.concat([tmptest,tmptrain],ignore_index=True)
        tmp['total']=abs(tmp['longitude']-lo)+abs(tmp['latitude']-la)
        idxmin=tmp['total'].idxmin()
        la=tmp['latitude'][idxmin]
        lo=tmp['longitude'][idxmin]
        for j in train.index[(abs(train['latitude']-la)<cut)&(abs(train['longitude']-lo)<cut)]:
            train['building_id_new'].values[j]=buid
        for j in test.index[(abs(test['latitude']-la)<cut)&(abs(test['longitude']-lo)<cut)]:
            test['building_id_new'].values[j]=buid
        #del train['tmplo']
    return train,test

def FillMissingIDNew(train,test,cut=0.0002):
    N=0
    while (len(train[train['building_id_new']=="0"])+len(test[test['building_id_new']=="0"]))>0:
        lo=0.0
        la=0.0
        if len(train[train['building_id_new']=="0"])>0:
            index=train.index[train['building_id_new']=="0"][0]
            lo=train['longitude'].values[index]
            la=train['latitude'].values[index]
        else:
            index=test.index[test['building_id_new']=="0"][0]
            lo=test['longitude'].values[index]
            la=test['latitude'].values[index]
        for j in train.index[(abs(train['latitude']-la)<cut)&(abs(train['longitude']-lo)<cut)]:
            train['building_id_new'].values[j]="Id_{}".format(N)
        for j in test.index[(abs(test['latitude']-la)<cut)&(abs(test['longitude']-lo)<cut)]:
            test['building_id_new'].values[j]="Id_{}".format(N)
        N=N+1
    return train,test

def MakeBinnig(train,test,column,binsize=0.001):
    tmp=pd.concat([test[column],train[column]],ignore_index=True)
    mean=tmp.mean()
    train["{}_bins".format(column)]=train[column].apply(lambda x:int((x-mean)/binsize))
    test["{}_bins".format(column)]=test[column].apply(lambda x:int((x-mean)/binsize))
    return train ,test
def AddColumnEmailPhone(df):
    df['phone_email']=0
    for j in range(len(df)):
        i=df['description'].values[j]
        i=i.lower()
        if i.find('call')>-1 or i.find('kagglemanager@renthop.com')>-1:
            df['phone_email'].values[j]=1
    return df
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


def transform_data(X,global_prob,feature_transform,flagtrain=True,cut=-1.0):
    #add features
    X['lenf']=X["features"].apply(len)
    mergedict=dict()

    mergedict["Doorman"]=['24_hour_doorman','24hr_doorman','7_doorman','7_doorman_concierge','_doorman','_doorman_','doorman',
     'four_hour_concierge_and_doorman','ft_doorman','hour_concierge_and_doorman','hour_doorman','time_doorman',
       '24_hour_concierge','7_concierge','concierge','concierge_service']
    mergedict["Laundry"]=['_laundry','laundry','laundry_','laundry_in_building','laundry_in_unit','laundry_on_every_floor',
    'laundry_on_floor','laundry_room','private_laundry_room_on_every_floor','site_laundry','valet_laundry']

    mergedict["Pets"]=['_pets_ok_','all_pets_ok','no_pets','pets','pets_','pets_allowed',
    'pets_ok','pets_on_approval','dogs_allowed','_cats','_cats_ok_','cats_allowed']

    mergedict["Nofee"]=["_diamond_no_fee_deal","_no_fee","fee","low_fee",
                   "no_broker_fee","no_fee","one_month_fee","reduced_fee"]

    mergedict["Balcony"]=["balcony","common_balcony","private_balcony"]

    mergedict["HFloor"]=["hardwood_floor","hardwood_flooring","hardwood_floors","hardwood","oak_floors"]

    mergedict["Dishwasher"]=["_dishwasher","_dishwasher_","dishwasher"]

    mergedict["Elevator"]=["elevator","elevator_"]

    mergedict["PreWar"]=["pre","pre_war","prewar"]
    mergedict["PostWar"]=["post","pre_war","prewar"]

    mergedict["Fitness"]=["art_cardio_and_fitness_club","art_fitness_center","equipped_club_fitness_center",
    "fitness","fitness_center","fitness_facility","fitness_room","gym"]

    mergedict["Outdoor"]=["common_outdoor_space","outdoor","outdoor_areas","outdoor_entertainment_space",
    "outdoor_roof_deck_overlooking_new_york_harbor_and_battery_park","outdoor_space"
    "outdoor_terrace","private_outdoor_space"]

    mergedict["Dining"]=["dining","dining_area","dining_room"]

    mergedict["RoofTerrace"]=["_roof_deck_","_scenic_roof_deck_","common_roof_deck",
    "outdoor_roof_deck_overlooking_new_york_harbor_and_battery_park",
    "roof","roof_access","roof_deck",
    "roof_deck_with_grills","roof_decks","roof_grilling_area","roofdeck","rooftop"
    "rooftop_deck","common_terrace","outdoor_terrace",
    "rooftop_terrace","terrace","terraces_","deck",
    "expansive_sundeck","sundeck"]


    mergedict["PRoofTerrace"]=["private_roof_access","private_roof_deck","private_roofdeck",
                               "_private_terrace_","private_terrace","private_deck"]

    mergedict["Pool"]=["indoor_swimming_pool","swimming_pool","indoor_pool","pool"]

    mergedict["ParkingGarage"]=["common_parking","garage_parking","parking"
    ,"parking_available","parking_garage","parking_space",
    "private_parking","site_parking","site_parking_available"
    "site_parking_lot","underground_parking","valet_parking",
    "full_service_garage","garage","site_attended_garage","site_garage"]

    mergedict["ParkingGarage"]=["_private_terrace_","private_terrace"]

    mergedict["HSinternet"]=["high_speed_internet","high_speed_internet_available","speed_internet"]

    mergedict["Wheelchair"]=["wheelchair_access","wheelchair_ramp"]

    mergedict["Photos"]=["_photos","_photos_","actual_photos"]


    mergedict["Garden"]=["common_garden","garden","gardening_area_with_rentable_plots","residents_garden","shared_garden"]

    mergedict["PrivateGarden"]=["_private_garden_","private_garden"]

    mergedict["Fireplace"]=["_fireplace_","burning_fireplace","deco_fireplace",
    "decorative_fireplace","fireplace","fireplaces",
    "working_fireplace"]

    mergedict["Ceiling"]=["high_ceiling","high_ceilings"]


    mergedict["Renovated"]=["_fully_renovated_","_gut_renovated_","fully_renovated","gut_renovated","newly_renovated","renovated"]
    mergedict["Washer"]=["in_unit_washer_and_dryer","unit_washer","washer","washer_","washer_in_unit"]

    mergedict["Lounge"]=["duplex_lounge","enclosed_private_lounge_with_magnificent_river_views","lounge","lounge_room","residents_lounge",
    "tenant_lounge"]
    mergedict["Patio"]=["patio","private_patio"]

    feat_sparse = feature_transform.transform(X["features"])
    vocabulary = feature_transform.vocabulary_
    del X['features']
    X1 = pd.DataFrame([ pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0]) ])
    X1.columns = list(sorted(vocabulary.keys()))

    if cut>0.0:
        for i in mergedict.keys():
            X1=mergecolumns(X1,mergedict[i],i)
        if flagtrain==True:
            X1=RemoveLowSatColumns(X1,cut)

    X = pd.concat([X.reset_index(), X1.reset_index()], axis = 1)
    del X['index']

    X["num_photos"] = X["photos"].apply(len)
    X['created'] = pd.to_datetime(X["created"])
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X['price_per_bed'] = X['price'] / X['bedrooms']
    X['price_per_room'] = X['price'] / (X['bathrooms'] + X['bedrooms'] )
    maxperroom=max(X[(X['bedrooms']!=0)|(X['bathrooms']!=0)]['price_per_room'])
    maxperbed=max(X[X['bedrooms']!=0]['price_per_room'])
    for i in X.index[(X['bedrooms']==0)&(X['bathrooms']==0)]:
        X['price_per_room'].values[i]=maxperroom
    for i in X.index[(X['bedrooms']==0)]:
        X['price_per_bed'].values[i]=maxperbed
    X["created"] = pd.to_datetime(X["created"])
    X["created_year"] = X["created"].dt.year
    X["created_month"] = X["created"].dt.month
    X["created_day"] = X["created"].dt.day
    X['created_hour'] = X["created"].dt.hour
    X['created_weekday'] = X['created'].dt.weekday
    X['created_week'] = X['created'].dt.week
    X['created_quarter'] = X['created'].dt.quarter
    #X['created_weekend'] = ((X['created_weekday'] == 5) & (X['created_weekday'] == 6))
    #X['created_wd'] = ((X['created_weekday'] != 5) & (X['created_weekday'] != 6))
    X["created_weekend"] = X['created_weekday'].apply(lambda x: 1 if ((x==5) | (x==6)) else 0)
    X["created_wd"] = X['created_weekday'].apply(lambda x: 1 if ((x!=5) & (x!=6)) else 0)
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
    X_train[['latitude','longitude']] = X_train[['latitude','longitude']].astype(float)
    X_test[['latitude','longitude']] = X_test[['latitude','longitude']].astype(float)
    image_date=LoadImgData()
    X_train = pd.merge(X_train, image_date, on="listing_id", how="left")
    X_test = pd.merge(X_test, image_date, on="listing_id", how="left")


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
    X_train = transform_data(X_train,global_prob,feature_transform,True,settings['others']["cut_on_cleaning_feauters"])
    X_test = transform_data(X_test,global_prob,feature_transform,False,settings['others']["cut_on_cleaning_feauters"])


    if settings['others']["clean_street_building_ids"]>0:
        print("Clean Street building 1")
        X_train=CreateSDN(X_train)
        X_test=CreateSDN(X_test)
        building_id_to_street,address_to_building_id,X_train,X_test=GetDict(X_train,X_test)

        if settings['others']["clean_street_building_ids"]>1:
            print("Clean Street building 2")
            X_train['building_id_new']=X_train['building_id']
            X_train['street_address_new_new']=X_train['street_address_new']
            X_test['building_id_new']=X_test['building_id']
            X_test['street_address_new_new']=X_test['street_address_new']
            if settings['others']["clean_street_building_ids"]==2:
                X_train=CleanBuildingID(X_train,address_to_building_id)
                X_test=CleanBuildingID(X_test,address_to_building_id)
                X_train,X_test=FillMissingID(X_train,X_test)
            if settings['others']["clean_street_building_ids"]==3:
                X_train=CleanStreet(X_train,building_id_to_street)
                X_test=CleanStreet(X_test,building_id_to_street)
            if settings['others']["clean_street_building_ids"]==4:
                print("Clean Street building 4")
                X_train=CleanBuildingID(X_train,address_to_building_id)
                X_test=CleanBuildingID(X_test,address_to_building_id)
                X_train=CleanStreet(X_train,building_id_to_street)
                X_test=CleanStreet(X_test,building_id_to_street)
                X_train,X_test=FillMissingID(X_train,X_test);
            if settings['others']["clean_street_building_ids"]==5:
                print("new way")
                X_train,X_test=NewWay(X_train,X_test,settings['others']["cut_lan_log_selection"])
                X_train,X_test=FillMissingIDNew(X_train,X_test,settings['others']["cut_lan_log_selection"])
            X_train['building_id']=X_train['building_id_new']
            X_test['building_id']=X_test['building_id_new']
            X_train['street_address']=X_train['street_address_new_new']
            X_test['street_address']=X_test['street_address_new_new']

    print("Normalizing high cordiality data...")
    for i in ['manager_id','street_address','building_id']:
        flag=settings['howtouseID'][i]
        if flag==1:
            final,labels=GetFractionsforIDStreet(X_train,i,settings['maxstat'][i],settings['others']["withrest"])
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
    if settings['others']['binsize']>0.0:
        X_train,Xtest=MakeBinnig(X_train,X_test,'longitude',settings['others']['binsize'])
        X_train,Xtest=MakeBinnig(X_train,X_test,'latitude',settings['others']['binsize'])

    X_train=AddColumnEmailPhone(X_train)
    X_test=AddColumnEmailPhone(X_test)
    if len(settings['others']['addNNresults'])>0:
        if len(settings['others']['diraddNNresults'])>0:
            trainNN= pd.read_csv("{}/train{}.csv".format(settings['others']['diraddNNresults'],settings['others']['addNNresults']))
            testNN= pd.read_csv("{}/test{}.csv".format(settings['others']['diraddNNresults'],settings['others']['addNNresults']))
        else:
            trainNN= pd.read_csv("./testtf/traintf{}.csv".format(settings['others']['addNNresults']))
            testNN= pd.read_csv("./testtf/testtf{}.csv".format(settings['others']['addNNresults']))
        tmpNN=pd.concat([trainNN,testNN])
        tmpNN.rename(columns={'high': 'NNhigh', 'medium': 'NNmedium',"low":"NNlow"}, inplace=True)
        X_train = pd.merge(X_train, tmpNN, on="listing_id", how="left")
        X_test = pd.merge(X_test, tmpNN, on="listing_id", how="left")
    return X_train,X_test

def LoadImgData():
    image_date = pd.read_csv("listing_image_time.csv")

    # rename columns so you can join tables later on
    image_date.columns = ["listing_id", "time_stamp"]

    # reassign the only one timestamp from April, all others from Oct/Nov
    image_date.loc[80240,"time_stamp"] = 1478129766

    image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
    image_date["img_month"]            = image_date["img_date"].dt.month
    image_date["img_week"]             = image_date["img_date"].dt.week
    image_date["img_day"]              = image_date["img_date"].dt.day
    image_date["img_weekday"]        = image_date["img_date"].dt.dayofweek
    image_date["img_dayofyear"]        = image_date["img_date"].dt.dayofyear
    image_date["img_hour"]             = image_date["img_date"].dt.hour
    image_date["img_monthBeginMidEnd"] = image_date["img_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)
    return image_date
def Cleanoutlayers(df1,df2,column):
    tmp=pd.concat([df1[column],df2[column]])
    std=10.0
    stdnew=1.0
    N=0
    while stdnew<std*0.95:
        N=N+1
        mean=tmp.mean()
        std=tmp.std()+1e-8
        for df in [df1,df2]:
            for i in range(len(df)):
                if df[column].values[i]<mean-5*std:
                    df[column].values[i]=mean-5*std
                if df[column].values[i]>mean+5*std:
                    df[column].values[i]=mean+5*std
        tmp=pd.concat([df1[column],df2[column]])
        stdnew=tmp.std()+1e-8
    return df1,df2

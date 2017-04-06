import operator
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import math
import re
import sys

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
    if "interest_level" in df_test.columns:
        maping_intrest={'low':0,'medium':1,'high':2}
        df_test['interest_level'] =df_test['interest_level'].map(maping_intrest)
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

def Std(df,columns):
    for i in columns:
        mean=df[i].mean();
        std=df[i].std()+1e-8
        df[i]=(df[i]-mean)/std
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

def GetDict(test):
    k=dict()
    p=dict()
    for i in test["building_id"].unique():
        tmp=test.loc[test["building_id"]==i]
        #print("{} {}".format(len(tmp),len(tmp['street_address_new'].unique())))
        tmplen=[len(j) if j!=' st' else 999 for j in tmp['street_address_new'].unique()]
        value=tmp['street_address_new'].unique()[tmplen.index(min(tmplen))]
        #m = re.search("[a-z]", value)
        m=value.find(" ")
        p[value]=i
        k[i]=value[m+1:]

    return k,p

def CleanIDstreet(df_train,train,train2):
    for i in range(len(df_train)):
        #if df_train['building_id'].values[i]=='0':
        if df_train['street_address_new'].values[i] in train2.keys():
            df_train['building_id_new'].values[i]=train2[df_train['street_address_new'].values[i]]
        if df_train['building_id_new'].values[i]!='0':
            if df_train['building_id_new'].values[i] in train.keys():
                df_train['street_address_new_new'].values[i]=train[df_train['building_id_new'].values[i]]
    return  df_train

def AddColumns(train,labels,column):
    for il,label in enumerate(labels):
        if len(label)>1: # chek not zero
            train["{}_{}".format(column,il)]= train[column].apply(lambda x: 1 if x==label else 0)


def ReadIn(filename,allparams):
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            line=line.replace("\n","")
            if line[0]=="#":
                print(line)
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
                            allparams[names[0]][names[1]]=type(allparams[names[0]][names[1]])(words[1])
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

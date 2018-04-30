import pandas as pd

from sklearn.model_selection import train_test_split

train = pd.read_csv('train_dataset.csv', header=0)

train = train.drop(axis=1,
                   labels=['age_as_on', 'Year_of_Revision', 'Guardian_gender', 'Voter_age', 'Voter_gender', 'index',
                           'Assembly_Constituency_name', 'House'])


labels = train.religion
del train['religion']
X_train,X_test,Y_train,Y_test=train_test_split(train,labels,test_size=0.25,shuffle=False)


# X_train=train
# Y_train=labels
# X_test=test
X_test = pd.DataFrame(X_test)
X_test = X_test.reset_index(drop=True)
Y_test = pd.DataFrame(Y_test)
Y_train=pd.DataFrame(Y_train)
Y_train=Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

names = pd.concat([X_train['Guardian_name'], X_train['Voter_name']], ignore_index=True)

print(names.head())
namelist = []
for name in names:
    namelist.append(name)

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer='char', ngram_range=(1,2))
vec.fit(namelist)
ft = vec.get_feature_names()
x = vec.transform(X_train['Guardian_name'])
print(type(x))
# r=pca.transform(x)
x = pd.DataFrame(x.todense(), columns=ft)
del X_train['Guardian_name']
X_train = X_train.join(x)

x = vec.transform(X_train['Voter_name'])
x = pd.DataFrame(x.todense(), columns=ft)
del X_train['Voter_name']
X_train = X_train.join(x, rsuffix='2', lsuffix='1')
X_train.fillna(0).to_sparse(fill_value=0)

x = vec.transform(X_test['Guardian_name'])
x = pd.DataFrame(x.todense(), columns=ft)
print(x)
del X_test['Guardian_name']
X_test = X_test.join(x, )
print(X_test.head())
X_test.fillna(0).to_sparse(fill_value=0)
print(X_test.head())

x = vec.transform(X_test['Voter_name'])
# r=pca.transform(x)
x = pd.DataFrame(x.todense(), columns=ft)
del X_test['Voter_name']
X_test = X_test.join(x, rsuffix='2', lsuffix='1')
X_test.fillna(0).to_sparse(fill_value=0)

from sklearn.decomposition import PCA

# pca=PCA(n_components=140)
# pca.fit(X_train)
# X_train=pca.transform(X_train)
# X_test=pca.transform(X_test)
# from sklearn.feature_selection import SelectPercentile,f_classif
# percentile = SelectPercentile(percentile=20)
#
# X_train = percentile.fit_transform(X_train, Y_train)
# selected_features = percentile.get_support(True)
#
# X_test = X_test.iloc[:, selected_features]
# print(selected_features)


# import xgboost as xgb

# mod = xgb.XGBClassifier()
# mod.fit(X_train, Y_train)
# print('xgb', mod.score(X_test, Y_test))
# exit(0)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

mod =AdaBoostClassifier()
mod.fit(X_train,Y_train)
print("ada",mod.score(X_test,Y_test))

mod = RandomForestClassifier()
mod.fit(X_train,Y_train)
print("fore",mod.score(X_test,Y_test))
# exit(0)

from sklearn.tree import DecisionTreeClassifier

mod = DecisionTreeClassifier()
mod.fit(X_train, Y_train)
print('dt', mod.score(X_test, Y_test))

from sklearn.naive_bayes import GaussianNB

mod = GaussianNB()
mod.fit(X_train, Y_train)
print('nb', mod.score(X_test, Y_test))

from sklearn.svm import SVC
import numpy as np
mod = SVC(C=1, kernel='rbf')
mod.fit(X_train,Y_train)
ypred=mod.predict(X_test)
print(ypred[0:20],Y_test[0:20])
print(mod.score(X_test,Y_test))
# import pickle
# filename='svmmodel.sav'
# pickle.dump(mod,open(filename,'wb'))
# y_pred =mod.predict(X_test)
# print(y_pred[0:20])
# cs=np.concatenate((X_test,y_pred),axis=1)
# cs=pd.DataFrame(cs)
# cs.to_csv("predictions.csv")

# X=np.concatenate((X_train,X_test),axis=0)
# print(Y_train.shape,Y_test.shape)
# print(Y_train[0:5],Y_test[0:5])
# Y=np.concatenate((Y_train,Y_test),axis=0)
# from sklearn.model_selection import cross_val_score
# score= cross_val_score(mod,X,Y,cv=10)
# print(score,score.mean(),score.std())

# mod.fit(X_train, Y_train)
#
print('svm', mod.score(X_test, Y_test))




# wordvect= Word2Vec(namelist,min_count=1,size=100)
# words=list(wordvect.wv.vocab)
# #print(wordvect['jchandrashekar rao  jasti'])
# vectors=wordvect[wordvect.wv.vocab]
# # from sklearn.decomposition import PCA
# #
# # pca= PCA(n_components=10)
# # pca=pca.fit(vectors)
# # print(pca.explained_variance_ratio_)
#
# x=wordvect[X_train['Guardian_name']]
# print(X_train.head())
# # r=pca.transform(x)
# x =pd.DataFrame(x)
# del X_train['Guardian_name']
# X_train=X_train.join(x)
#
# x=wordvect[X_train['Voter_name']]
# # r=pca.transform(x)
# x=pd.DataFrame(x)
# del X_train['Voter_name']
# X_train=X_train.join(x,rsuffix='2',lsuffix='1')
#
# print(X_train.head())


#
# for name in X_train['Guardian_name']:
#     x=wordvect[name]
#     r=pca.transform(x)

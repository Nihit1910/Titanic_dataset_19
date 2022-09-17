import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('train.csv')
print(dataset.shape)
dataset['Age']=dataset['Age'].fillna(-1)
dataset['Embarked'].fillna('S', inplace=True)
X=dataset.iloc[:, [2,4,5,6,7,11]].values
y=dataset.iloc[:, 1].values


from sklearn.impute import SimpleImputer
imputer_age= SimpleImputer(missing_values=-1, strategy="median")
imputer_age=imputer_age.fit(X[:, [2]])
X[:, [2]]=imputer_age.transform(X[:, [2]])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_sex = LabelEncoder()
X[:, 1] = labelencoder_sex.fit_transform(X[:, 1])
labelencoder_embark = LabelEncoder()
X[:, 5] = labelencoder_embark.fit_transform(X[:, 5])
ct = ColumnTransformer([("embark", OneHotEncoder(), [5])], remainder = 'passthrough')
X = ct.fit_transform(X)

X=X[:,1:]

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)
logreg.fit(X_train,y_train)

y_predlogreg=logreg.predict(X_test)
accuracies1=cross_val_score(estimator=logreg,X=X_train,y=y_train,cv=10)
accuracylogreg=accuracies1.mean()*100
                       
from sklearn.ensemble import RandomForestClassifier
model_random = RandomForestClassifier(n_estimators = 1000,oob_score=True,random_state=0,min_samples_split=10)
model_random.fit(X_train,y_train)
y_predrandom=model_random.predict(X_test)
accuracies2=cross_val_score(estimator=model_random,X=X_train,y=y_train,cv=10)
accuracyrandom=accuracies2.mean()*100

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predsvc=svc_model.predict(X_test)
accuracies3=cross_val_score(estimator=svc_model,X=X_train,y=y_train,cv=10)
accuracysvc=accuracies3.mean()*100
                       
from sklearn.neighbors import KNeighborsClassifier
knn_model  = KNeighborsClassifier(n_neighbors = 14)
knn_model.fit(X_train, y_train)
y_predknn=knn_model.predict(X_test)
accuracies4=cross_val_score(estimator=knn_model,X=X_train,y=y_train,cv=10)
accuracyknn=accuracies4.mean()*100

from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
model_naive.fit(X_train, y_train)
y_prednaive=model_naive.predict(X_test)
accuracies5=cross_val_score(estimator=model_naive,X=X_train,y=y_train,cv=10)
accuracynaive=accuracies5.mean()*100
  
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini',min_samples_split=12,max_features='sqrt',min_samples_leaf=12)
tree_model.fit(X_train, y_train)
y_predtree=tree_model.predict(X_test)
accuracies6=cross_val_score(estimator=tree_model,X=X_train,y=y_train,cv=10)
accuracytree=accuracies6.mean()*100

print({'accuracy_logreg':accuracylogreg,'accuracyrandom':accuracyrandom,'accuracysvc':accuracysvc,'accuracyknn':accuracyknn,'accuracynaive':accuracynaive,'accuracytree':accuracytree})                    
                                              
dataset1=pd.read_csv('test.csv')
dataset1['Age']=dataset1['Age'].fillna(-1)
dataset1['Embarked'].fillna('S', inplace=True)
z=dataset1.iloc[:, [1,3,4,5,6,10]].values

from sklearn.impute import SimpleImputer
imputer_age= SimpleImputer(missing_values=-1, strategy="median")
imputer_age=imputer_age.fit(z[:, [2]])
z[:, [2]]=imputer_age.transform(z[:, [2]])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_sex = LabelEncoder()
z[:, 1] = labelencoder_sex.fit_transform(z[:, 1])
labelencoder_embark = LabelEncoder()
z[:, 5] = labelencoder_embark.fit_transform(z[:, 5])
ct = ColumnTransformer([("embark", OneHotEncoder(), [5])], remainder = 'passthrough')
z = ct.fit_transform(z)

z=z[:,1:]

from sklearn.preprocessing import StandardScaler
sc_z=StandardScaler()
z=sc_z.fit_transform(z)

y_pred=svc_model.predict(z)

submission = pd.DataFrame({
        "PassengerId": dataset1["PassengerId"],
        "Survived": y_pred})

submission.to_csv("submission.csv",index=False)
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('diabetes.csv')

'''make report for visualize and stat'''
# profile=ProfileReport(data,)
# profile.to_file('report.html')
X=data.drop('Outcome',axis=1)
y=data['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=159)



scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

params={'n_estimators' :[50,100,150],
         'max_depth'   :[1,2,3,5,7],
         'max_features':['sqrt','log2'],  
         'criterion'   : ['gini','entropy','log_loss']         
}


model=RandomForestClassifier()
clf=GridSearchCV(model,params)
clf.fit(X_train,y_train)
print(clf.best_params_)


predicted_val=clf.predict(X_test)
print(classification_report(y_test,predicted_val))

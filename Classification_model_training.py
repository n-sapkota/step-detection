# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc,roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector


#Load dataset as pandas data frame
dataset = pd.read_csv("centre_both.csv")

#Split data into input and output variable
X = dataset.iloc[:,0:dataset.shape[1]-1]
Y = dataset.iloc[:,-1]

# select 70% instances to train the classifier
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# Removing Constant features
constant_filter = VarianceThreshold()
constant_filter.fit(X_train)
constant_columns = [col for col in X_train.columns  
                    if col not in X_train.columns[constant_filter.get_support()]]
X_train.drop(labels=constant_columns, axis=1, inplace=True) 
X_test.drop(labels=constant_columns, axis=1, inplace=True)


start = time.time()
classifier_ = DecisionTreeClassifier(random_state = 100) 
selector = RFE(classifier_, 5, step=1)
selector = selector.fit(X_train, y_train)
end = time.time()
print("Execution time: %0.4f seconds"%(float(end)- float(start)))
unselected_features = [column for column in X_train.columns if not column in X_train.columns[selector.support_]]

X_train.drop(labels = unselected_features,axis =1,inplace =True )
X_test.drop(labels = unselected_features,axis =1,inplace =True )


start = time.time()
clf3 = DecisionTreeClassifier(random_state = 100)
clf3.fit(X_train, y_train)
end = time.time()

y_pred = clf3.predict(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("Size of Data set : %.2f MB"%(X_train.values.nbytes/1e6))
print('Accuracy :%0.5f' %accuracy_score(y_test, y_pred))
print('F1 Score :%0.5f'%f1_score(y_test, y_pred))
print('Area under ROC : %0.5f'%roc_auc)
print("Execution time for training Decision tree : %f seconds"%(float(end)- float(start)))

train_pred = clf3.predict_proba(X_train)  
print('Accuracy on training set: %.2f'%(roc_auc_score(y_train, train_pred[:,1])))
test_pred = clf3.predict_proba(X_test)  
print('Accuracy on test set: {}'.format(roc_auc_score(y_test, test_pred [:,1]))) 


#%%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc,roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector


#Load dataset as pandas data frame
dataset = pd.read_csv("centre_both.csv")

#Split data into input and output variable
X = dataset.iloc[:,0:dataset.shape[1]-1]
Y = dataset.iloc[:,-1]

# select 70% instances to train the classifier
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# Removing Constant features
constant_filter = VarianceThreshold()
constant_filter.fit(X_train)
constant_columns = [col for col in X_train.columns  
                    if col not in X_train.columns[constant_filter.get_support()]]
X_train.drop(labels=constant_columns, axis=1, inplace=True) 
X_test.drop(labels=constant_columns, axis=1, inplace=True)

from sklearn.neighbors import KNeighborsClassifier
start = time.time()
classifier_ = DecisionTreeClassifier(random_state = 100)
knn = KNeighborsClassifier(n_neighbors=2) 
feature_selector = SequentialFeatureSelector(classifier_,  
           k_features=15,
           forward=True,
           scoring='accuracy',
           cv=0)
feature_selector = feature_selector.fit(X_train,y_train)
end = time.time()
print("Execution time: %0.4f seconds"%(float(end)- float(start)))
selected_features= X_train.columns[list(feature_selector.k_feature_idx_)]

X_train = X_train[selected_features]
X_test = X_test[selected_features]


start = time.time()
clf3 = DecisionTreeClassifier(random_state = 100)
clf3.fit(X_train, y_train)
end = time.time()

y_pred = clf3.predict(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("Size of Data set : %.2f MB"%(X_train.values.nbytes/1e6))
print('Accuracy :%0.5f' %accuracy_score(y_test, y_pred))
print('F1 Score :%0.5f'%f1_score(y_test, y_pred))
print('Area under ROC : %0.5f'%roc_auc)
print("Execution time for training Decision tree : %f seconds"%(float(end)- float(start)))

train_pred = clf3.predict_proba(X_train)  
print('Accuracy on training set: %0.4f'%(roc_auc_score(y_train, train_pred[:,1])))
test_pred = clf3.predict_proba(X_test)  
print('Accuracy on test set: %0.4f'%(roc_auc_score(y_test, test_pred [:,1]))) 


#%%

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import wine_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
#Load dataset as pandas data frame
dataset = pd.read_csv("centre_both.csv")

#Split data into input and output variable
X = dataset.iloc[:,0:dataset.shape[1]-1]
Y = dataset.iloc[:,-1]
X_train, X_test, y_train, y_test= train_test_split(X, Y,
                                                   stratify=Y,
                                                   test_size=0.3,random_state = 100)


# Removing Constant features
constant_filter = VarianceThreshold()
constant_filter.fit(X_train)
constant_columns = [col for col in X_train.columns
                    if col not in X_train.columns[constant_filter.get_support()]]
X_train.drop(labels=constant_columns,axis=1, inplace=True) 
X_test.drop(labels=constant_columns,axis=1, inplace=True)
classifier_ = DecisionTreeClassifier(random_state = 100)

sfs1 = SFS(estimator=classifier_, 
           k_features=(5, 30),
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=3)

pipe = make_pipeline(StandardScaler(), sfs1)

pipe.fit(X_train, y_train)

print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
plot_sfs(sfs1.get_metric_dict(), kind='std_err');



selected_features1 = list(sfs1.k_feature_names_)
# save the model to disk
model = LogisticRegression()
model.fit(X_train, Y_train)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)

#%%
# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
#Load dataset as pandas data frame
dataset = pd.read_csv("centre_both.csv")

#Split data into input and output variable
X = dataset.iloc[:,0:dataset.shape[1]-1]
Y = dataset.iloc[:,-1]
X_train, X_test, y_train, y_test= train_test_split(X, Y,
                                                   stratify=Y,
                                                   test_size=0.3,random_state = 100)



# Removing Constant features
constant_filter = VarianceThreshold()
constant_filter.fit(X_train)
constant_columns = [col for col in X_train.columns
                    if col not in X_train.columns[constant_filter.get_support()]]
X_train.drop(labels=constant_columns,axis=1, inplace=True) 
X_test.drop(labels=constant_columns,axis=1, inplace=True)
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
# display the relative importance of each attribute
print(model.feature_importances_)


print('Feature_Name','\t\t','Ranking')
for feature_name, feature_score in zip(X_train.columns,model.feature_importances_):
    print(feature_name, '\t\t', feature_score)

scores = list(model.feature_importances_)

df = pd.DataFrame()
df['Feature'] = X_train.columns
df['value'] =scores 
ndf = df.sort_values(by=['value'],ascending=False)

def print_best_worst (scores):
    scores = sorted(scores, reverse = True)
    
    print("The 5 best features selected by this method are :")
    for i in range(5):
        print(scores[i])
    
    print ("The 5 worst features selected by this method are :")
    for i in range(5):
        print(scores[len(scores)-1-i])

print_best_worst (scores)
#%%
print('======= BEFORE FEATURE SELECTION ========')
# training decision tree
start = time.time()
clf = DecisionTreeClassifier(random_state = 100)
clf.fit(X_train, y_train)
end = time.time()

#y_pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)
y_pred = probs[:,1]
fpr1, tpr1, threshold = roc_curve(y_test, y_pred)
roc_auc1 = auc(fpr1, tpr1)

print("Size of Data set : %.2f MB"%(X_train.values.nbytes/1e6))
print('Accuracy :%0.5f' %accuracy_score(y_test, y_pred))
print('F1 Score :%0.5f'%f1_score(y_test, y_pred))
print('Area under ROC : %0.5f'%roc_auc1)
print("Execution time for training Decision tree : %f seconds"%(float(end)- float(start)))

plt.title('Before feature selection')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.5f' % roc_auc1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.09, 1.05])
plt.ylim([-0.09, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.save('thisistest.png')

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc,roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector


#Load dataset as pandas data frame
dataset = pd.read_csv("centre_both.csv")

#Split data into input and output variable
X = dataset.iloc[:,0:dataset.shape[1]-1]
Y = dataset.iloc[:,-1]

# select 70% instances to train the classifier
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# Removing Constant features
constant_filter = VarianceThreshold()
constant_filter.fit(X_train)
constant_columns = [col for col in X_train.columns  
                    if col not in X_train.columns[constant_filter.get_support()]]
X_train.drop(labels=constant_columns, axis=1, inplace=True) 
X_test.drop(labels=constant_columns, axis=1, inplace=True)


#
from sklearn import svm
model = svm.SVC(kernel='linear',gamma=1,probability=True)
model.fit(X_train, y_train)
train_pred = model.predict_proba(X_train)  
print('Accuracy on training set: %0.4f'%(roc_auc_score(y_train, train_pred[:,1])))
test_pred = model.predict_proba(X_test)  
print('Accuracy on test set: %0.4f'%(roc_auc_score(y_test, test_pred [:,1])))


#
print('\n')
clf3 = DecisionTreeClassifier(random_state = 100)
clf3.fit(X_train, y_train)
train_pred = clf3.predict_proba(X_train)  
print('Accuracy on training set: %0.4f'%(roc_auc_score(y_train, train_pred[:,1])))
test_pred = clf3.predict_proba(X_test)  
print('Accuracy on test set: %0.4f'%(roc_auc_score(y_test, test_pred [:,1])))




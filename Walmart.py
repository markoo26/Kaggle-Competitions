### IMPORT OF THE LIBRARIES ###

import pandas as pd, numpy as np, matplotlib as mpl
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

### Libraries for classifiers

### DATA PREPARATION

train_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\Walmart\\train.csv')
test_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\Walmart\\test.csv')

datasets = [train_data, test_data]

label_encoder = LabelEncoder()

DICT_weekend = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 
                'Thursday': 0, 'Friday': 1, 'Saturday': 1,
                'Sunday': 1}

DICT_weekday = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 
                'Thursday': 4, 'Friday': 5, 'Saturday': 6,
                'Sunday': 7}

for dataset in datasets:
    dataset['IsWeekend'] = dataset['Weekday'].map(DICT_weekend)
    dataset['Weekday'] = dataset['Weekday'].map(DICT_weekday)
    dataset['ProductsReturned'] = dataset['ScanCount'][dataset['ScanCount']<0]
    dataset['ProductsReturned'] = dataset['ProductsReturned'] * (-1)
    dataset['ProductsReturned'] = dataset['ProductsReturned'].fillna(value=0)
    dataset['ProductsPurchased'] = dataset['ScanCount'][dataset['ScanCount']>0]
    dataset['ProductsPurchased'] = dataset['ProductsPurchased'].fillna(value=0)

    single_visit = dataset.groupby('VisitNumber').agg({'VisitNumber': ['count']})
    single_visit = single_visit.set_index(single_visit.index).T.to_dict('records')    
    dataset['ProductsPerVisit'] = dataset['VisitNumber'].map(single_visit[0])
    dataset['DepartmentDescription'] = label_encoder.fit_transform(train_data.DepartmentDescription.values.astype(str))

train_columns = ['ProductsPurchased','ProductsReturned','Weekday', 'ProductsPerVisit', 'DepartmentDescription']

y = train_data['TripType']
X = train_data[train_columns]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

### Building Classifiers 

bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)

rfc_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
rfc_classifier.fit(X_train, y_train)

log_classifier = LogisticRegression(random_state = 0)
log_classifier.fit(X_train, y_train)

xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)

ann_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
ann_classifier.fit(X_train, y_train)

knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p = 2)
knn_classifier.fit(X_train, y_train)

svc_classifier = SVC(kernel = 'linear', random_state = 0)
svc_classifier.fit(X_train, y_train)

y_pred_bayes = bayes_classifier.predict(X_test)
y_pred_rfc = rfc_classifier.predict(X_test)
y_pred_log = log_classifier.predict(X_test)
y_pred_xgb = xgb_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_ann = ann_classifier.predict(X_test)
y_pred_svc = svc_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
bayes_cm = confusion_matrix(y_test, y_pred_bayes)
rfc_cm = confusion_matrix(y_test, y_pred_rfc)
log_cm = confusion_matrix(y_test, y_pred_log)
xgb_cm = confusion_matrix(y_test, y_pred_xgb)
knn_cm = confusion_matrix(y_test, y_pred_knn)
ann_cm = confusion_matrix(y_test, y_pred_ann)
svc_cm = confusion_matrix(y_test, y_pred_svc)

def accuracy(conf_matrix):
    positives = 0 
    for i in range(0,38):
        positives = positives + conf_matrix[i][i]
        value =  positives/ sum(sum(conf_matrix))
    return "{0:.2f}%".format(value * 100)

print('Bayes: ' + str(accuracy(bayes_cm)))
print('Random Forest: ' + str(accuracy(rfc_cm)))
print('Logistic Regression: ' + str(accuracy(log_cm)))
print('XG-Boost: ' + str(accuracy(xgb_cm)))
print('K-nearest Neighbors: ' + str(accuracy(knn_cm)))
print('Artificial Neural Network: ' + str(accuracy(ann_cm)))
print('SVC: ' + str(accuracy(svc_cm)))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_bayes = cross_val_score(estimator = bayes_classifier, X = X_train, y = y_train, cv = 2)
accuracies_rfc = cross_val_score(estimator = rfc_classifier, X = X_train, y = y_train, cv = 2)
accuracies_log = cross_val_score(estimator = log_classifier, X = X_train, y = y_train, cv = 2)
accuracies_xgb = cross_val_score(estimator = xgb_classifier, X = X_train, y = y_train, cv = 2)
accuracies_knn = cross_val_score(estimator = knn_classifier, X = X_train, y = y_train, cv = 2)
accuracies_ann = cross_val_score(estimator = ann_classifier, X = X_train, y = y_train, cv = 2)

accuracies_bayes
accuracies_rfc
accuracies_log
accuracies_xgb
accuracies_knn
accuracies_ann

code = str('knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = "euclidean", p = 2) \n') + str('knn_classifier.fit(X_train, y_train)')

def count_exec_time(cmd_string, title = None):
    start_time = time.time()
    exec(cmd_string)
    end_time = time.time()
    exec_time = end_time - start_time
    exec_summary = 'Execution time of: ' + title + ' was ' + str(round(exec_time,2)) + ' seconds'
    return exec_summary

count_exec_time(code, 'KNN')

with open('kodzik.txt', 'r') as file:
    walmart = file.read()
    
    count_exec_time(walmart, 'total code')

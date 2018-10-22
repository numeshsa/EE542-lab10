# importing necessary libraries
import pandas as pd 
import hashlib
import os 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split 

def lassoSelection(X_train, y_train, n):
    '''
    Lasso feature selection.  Select n features. 
    '''
    #lasso feature selection
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0.01)
    sfm.fit(X_train, y_train)
    X_transform = sfm.transform(X_train)
    n_features = X_transform.shape[1]
    
    while n_features > n:
        sfm.threshold += 0.01
        X_transform = sfm.transform(X_train)
        n_features = X_transform.shape[1]
    features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
    print("selected features are {}".format(features))
    return features

# loading the dataset 
data_dir ="/Users/nikhilumeshsargur/Downloads/lab10/data/"

data_file = data_dir + "miRNA_matrix.csv"

df = pd.read_csv(data_file)
# print(df)

df.label = pd.factorize(df.label)[0]
y_data = df.pop('label').values

df.pop('file_id')

#print (columns)
X_data = df.values 
  
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)

#standardize the data.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Percentage of tumor cases in training set is {}".format(sum(y_train)/len(y_train)))
print("Percentage of tumor cases in testing set is {}".format(sum(y_test)/len(y_test)))

n = 50
features_columns = lassoSelection(X_train, y_train, n)

scores={}
key = 'KNeighborsClassifier'
clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': list(range(1,11))}, scoring=None,  refit=True, cv=10)
clf.fit(X_train[:, features_columns],y_train)
y_test_predict = clf.predict(X_test[:, features_columns])
precision = precision_score(y_test, y_test_predict, average='micro')
accuracy = accuracy_score(y_test, y_test_predict)
f1 = f1_score(y_test, y_test_predict, average='micro')
recall = recall_score(y_test, y_test_predict, average='micro')
scores[key] = [precision,accuracy,f1,recall]

#Visulization using PCA and tSNE
X_data = scaler.transform(X_data)

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(X_data)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result_50)

plt.scatter(tsne_results[:,0],tsne_results[:,1], c=y_data, s=6)
plt.colorbar()
plt.show()

#Evaluation of the dataset
ax = plt.subplot(111)
precisions = []
accuracies =[]
f1_scores = []
recalls = []
categories = []
specificities = []
N = len(scores)
ind = np.arange(N)
width = 0.1
for key in scores:
	categories.append(key)
	precisions.append(scores[key][0])
	accuracies.append(scores[key][1])
	f1_scores.append(scores[key][2])
	recalls.append(scores[key][3])

precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')

print(categories)
ax.set_xticks(np.arange(N))
ax.set_xticklabels(categories)
ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0]), ('precision', 'accuracy','f1','sensitivity'))
ax.grid()
plt.show()

#ROC_CURVE
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predict, pos_label=10)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_CURVE')
plt.legend(loc="lower right")
plt.show()
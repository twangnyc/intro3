import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#%matplotlib inline

def createplot(clf, savename):
    fig = plt.figure()
    width=0.35
    ax = fig.add_subplot(111)
    ax.bar(np.arange(11), clf.feature_importances_, width, color='r')
    ax.set_xticks(np.arange(len(clf.feature_importances_)))
    ax.set_xticklabels(X_test.columns,rotation=90)
    plt.title('Feature Importance from DT')
    plt.savefig(savename+".png", dpi=120)


data = pd.read_csv('Cell2Cell_data.csv')
train_split = 0.8
train_index = random.sample(range(0,len(data)),int(len(data)*train_split))
test_index = list(set(range(0,len(data)))-set(train_index))
train_set = data.ix[train_index]
X_train = train_set[:]
del X_train['churndep']
Y_train = train_set['churndep']
test_set = data.ix[test_index]
X_test = test_set[:]
del X_test['churndep']
Y_test = test_set['churndep']

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,Y_train)
ans = clf.predict(X_test)
createplot(clf, 'test')

split = range(100,2000,50)
leaf = range(100,1000,50)
for i in split:
    for j in leaf:
        clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=i, min_samples_leaf=j)
        clf.fit(X_train,Y_train)
        ans = clf.predict(X_test)
        if float((len(ans)-sum(abs(ans-Y_test))))/len(ans)>0.59:
            print 'split = %d, leaf = %d' %(i,j)
            print float((len(ans)-sum(abs(ans-Y_test))))/len(ans)




#data_3most = data[['revenue','eqpdays','outcalls']]
#cor=data_3most.corr()
#print cor
#print float((len(ans)-sum(abs(ans-Y_test))))/len(ans)
#print clf.feature_importances_


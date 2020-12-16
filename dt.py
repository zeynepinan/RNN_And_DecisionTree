import pandas as pd

feature_names = ['region-centroid-col', 'region-centroid-row', 'region-pixel-count', 'short-line-density-5', 
                 'short-line-density-2', 'vedge-mean',  'vegde-sd', 'hedge-mean', 'hedge-sd', 'intensity-mean',
                 'rawred-mean','rawblue-mean','rawgreen-mean','. exred-mean','exblue-mean','exgreen-mean',
                 'value-mean','saturatoin-mean','hue-mean', 'class']
dataset=pd.read_csv("segment.dat", sep=' ', names=feature_names)
print(dataset)
label=dataset.iloc[:,-1:].values #-1 dediğimiz için son kolonu yani sınıf etiketlerini getirecektir..
print(label)
#veri kümesini train test olarak ayırmak için gerekli kütüphaneyi import edip içinden gerekli fonksiyonu 
#çağırıyoruz.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:,:-1],label, test_size=0.5, random_state=0) 


#artık sınıflandırma yapabiliriz. modeli öğreniyoruz;



from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion="entropy", max_depth=9, splitter='random', max_leaf_nodes=10, 
  min_samples_leaf=5) #budama işlemi ezberlemeyi önlemek için
model = dt.fit(x_train, y_train)
sonuc = dt.predict(x_test)








#https://statinfer.com/204-3-10-pruning-a-decision-tree-in-python/
#başarı oranı;
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, sonuc)
print("Accuracy: ",accuracy)




from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

max_depth = []
acc_gini = []
acc_entropy = []

for i in range(1,30):
 decision_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)
 decision_tree.fit(x_train, y_train)
 pred = decision_tree.predict(x_test)
 acc_gini.append(accuracy_score(y_test, pred))
 ####
 decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
 decision_tree.fit(x_train, y_train)
 pred = decision_tree.predict(x_test)
 acc_entropy.append(accuracy_score(y_test, pred))
 ####
 max_depth.append(i)
d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
 'acc_entropy':pd.Series(acc_entropy),
 'max_depth':pd.Series(max_depth)})
# visualizing changes in parameters
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xticks(np.arange(1, 30, 2))
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.grid()
plt.legend()
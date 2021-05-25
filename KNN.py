#K nearest neighbours implementation from scratch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def knn(query_example,x_train,y_train,k):
    distances_and_labels = []
    for index,example in enumerate(x_train):
        distance = (example-query_example)
        distance = np.norm(distance)
        label = y_train[index]
        distances_and_labels.append((distance,label))
        
    sorted_distances_and_indices = sorted(distances_and_indices)
    k_nearest_distances_and_indices = sorted_distances_and_indices[:k]
    k_nearest_labels = k_nearest_distances_and_indices[:,-1]
    pred_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return pred_label

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
y_pred = []
for query_example in x_test:
    pred_label = knn(query_example,x_train,y_train,k)
    y_pred.append(pred_label)
    
y_pred = np.array(y_pred)
print(y_pred)

accuracy = accuracy_score(y_pred,y_test)
print("Accuracy",accuracy_score)

#Knearest neighbours using sklearn
from sklearn.neighbours import KNeighboursClassifier
knn = KNeighbousClassifier(n_neighbours = 7)
knn.fit(x_train,y_train)
accuracy_score = knn.score(x_test,y_test)
print("Accuracy",accuracy_score)

#Finding the best k value
nearest_neighbours = np.arange(1,9)
train_accuracy = np.empty(len(nearest_neighbours))
test_accuracy = np.empty(len(nearest_neighbours))

for i,k in enumerate(nearest_neighbours):
    knn = KNeighboursClassifier(n_neighbours = k)
    knn.fit(x_train,y_train)
    
    temp_train_acc = knn.score(x_train,y_train)
    temp_test_acc = knn.score(x_test,y_test)
    train_accuracy[i] = temp_train_acc
    test_accuracy[i] = temp_test_acc
    
plt.plot(nearest_neighbours,train_accuracy,label = 'Training data accuracy')
plt.plot(nearest_neighbours,test_accuracy,label = 'Test data accuracy')
plt.legend()
plt.xlabel('num_neighbours')
plt.ylabel('Accuracy')
plt.show()

















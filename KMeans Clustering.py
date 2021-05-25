import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm

path = 
data = pd.read_csv(path)
df.head()

#KMeans using self built function
class KMeans:
    def __init__(self,num_clusters,max_iter):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        
    def compute_initial_centroids(self,data):
        num_points = data.shape[0]
        indices = np.arange(num_points)
        np.random.shuffle(indices)
        centroids = data[indices[:self.num_clusters]]
        return centroids
    
    def compute_distance(self,data,centroids):
        num_points = data.shape[0]
        distance_matrix = np.zeros(num_points,self.num_clusters)
        for i in range(num_points):
            single_point = data[i,:]
            temp_dist = centroids-single_point
            temp_dist = norm(temp_dist)
            distance[i,:] = temp_dist
        
        return distance
    
    
    def calculate_labels(self,distance):
        labels = np.argmin(distance,axis=1)
        return labels
    
    def calculate_centroids(self,data,labels):
        num_vars = data.shape[1]
        centroids = np.zeros(self.num_clusters,num_vars)
        for k in range(self.num_clusters):
            single_centroid = np.mean(data[labels==k,:],axis=0)
            centroids[k] = single_centroid
            
        return centroids
    
    def compute_loss(self,data,labels,centroids):
        loss = 0
        num_points = data.shape[0]
        for i in range(num_points):
            single_point = data[i,:]
            point_centroid = centroids[labels[i],:]
            temp_dist = single_point-point_centroid
            temp_dist = norm(temp_dist)
            temp_dist = np.square(temp_dist)
            loss += temp_dist
            
        return loss
    
    def fit(self,data):
        self.centroids = self.compute_initial_centroids(data)
        for itr in range(max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(data,centroids)
            self.labels = self.calculate_labels(distance)
            self.centroids = self.calculate_centroids(data,self.labels)
            if np.all(old_centroids==self.centroids):
                break
        
    def predict(self,data,old_centroids):
        min_distance = self.compute_distance(data,old_centroids)
        final_clusters = self.calcualte_labels(distance)
        return final_clusters
    


#KMeans using sklearn 
data = StandardScaler().fit_transform(data)
x = data['Variable_1']
y = data['Variable_2']
fig,ax = plt.subplots()
ax.scatter(x,y,marker = 'o',c = 'red',alpha=0.5)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.show()

from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data)

labels = kmeans.predict(data)
print(labels)
centroids = kmeans.centroids
print(centroids)



        
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
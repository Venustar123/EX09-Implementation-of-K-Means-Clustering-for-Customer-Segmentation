# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Initialize K cluster centroids randomly.

2.Assign each data point to the nearest centroid, forming K clusters.

3.Recompute the centroids by calculating the mean of all data points in each cluster.

4.Repeat steps 2 and 3 until the centroids no longer change significantly (or a maximum iteration count is reached).

5.Output the clusters and their centroids.
## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: BASHA VENU
RegisterNumber:  2305001005

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
dt=pd.read_csv("/content/Mall_Customers_EX8.csv")
dt
x=dt[['Annual Income (k$)','Spending Score (1-100)']]
plt.figure(figsize=(4,4))
plt.scatter(dt['Annual Income (k$)'],dt['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(x)
centroids=Kmeans.cluster_centers_
labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```

## Output:
![WhatsApp Image 2024-11-06 at 21 41 33_c6b35970](https://github.com/user-attachments/assets/78008171-ac9e-4d82-b52d-a7a6cfb8601a)
![WhatsApp Image 2024-11-06 at 21 42 03_ca29ec4d](https://github.com/user-attachments/assets/0eaa12e7-b6be-4836-ba1f-d6c225797b7e)
![WhatsApp Image 2024-11-06 at 21 42 23_c59bef26](https://github.com/user-attachments/assets/236045a2-d251-4feb-99e2-9cc86a475750)
![WhatsApp Image 2024-11-06 at 21 42 47_6ce2c622](https://github.com/user-attachments/assets/9cc73c53-08aa-4f0d-bc9e-b52dc401177b)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

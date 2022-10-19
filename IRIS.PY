#Important library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
df=pd.read_csv('Iris.csv')

print(df.head(20))
print (df.shape)
print (df.describe())
#Training data 
x = df.iloc[:, [0,1,2,3]].values
#clustering using k=5.
kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)
kmeans5.cluster_centers_
#Elbow method
Error =[]
for i in range(1, 11):
 kmeans = KMeans(n_clusters = i).fit(x)    
 kmeans.fit(x)    
 Error.append(kmeans.inertia_)
 plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
#clustering using k=3
kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)
kmeans3.cluster_centers_
#Visualization of data
# k=5
plt.scatter(x[:,0],x[:,1],c=y_kmeans5,cmap='rainbow')
#Visualization of data
# k=3
plt.scatter(x[:,0],x[:,1],c=y_kmeans3,cmap='gist_rainbow')

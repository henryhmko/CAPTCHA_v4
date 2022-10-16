# from INIT,py import *
from setence_clustering import *



#T-SNE Dimension Reduction & Visualization
tsne = TSNE(n_components=2, verbose=1, random_state=123)
df = tsne.fit_transform(data)
kmeans = KMeans(n_clusters=6)
label = kmeans.fit_predict(df)

filtered_label0 = df[label == 0]
filtered_label1 = df[label == 1]
filtered_label2 = df[label == 2]
filtered_label3 = df[label == 3]
filtered_label4 = df[label == 4]
filtered_label5 = df[label == 5]

plt.scatter(filtered_label0[:,0] , filtered_label0[:,1], color = 'red')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1], color = 'purple')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1], color = 'yellow')
plt.scatter(filtered_label3[:,0] , filtered_label3[:,1], color = 'black')
plt.scatter(filtered_label4[:,0] , filtered_label4[:,1], color = 'green')
plt.scatter(filtered_label5[:,0] , filtered_label5[:,1], color = 'orange')

##UNCOMMENT LIINE BELOW TO VISUALIZE.
#plt.show()
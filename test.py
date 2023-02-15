import common
import em
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('Fire Station Database.csv')
df_values = df.iloc[:,8:].fillna(0).to_numpy()
X = df_values

mixture, post = common.init(X,3)

post, cost = em.estep(X, mixture)

mixture = em.mstep(X, post, mixture)

mixture, post, cost = em.run(X, mixture, post)

X = em.fill_matrix(X, mixture)

X = StandardScaler().fit_transform(X)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12))

visualizer.fit(X)
visualizer.show()

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

labels = kmeans.labels_

n_components = 3

transformer = PCA(n_components=n_components)

transformer.fit(X)

X_transformed = transformer.transform(X)

fig, axes = plt.subplots()
ax = plt.axes(projection='3d')

xdata = X_transformed[:,0]
ydata = X_transformed[:,1]
zdata = X_transformed[:,2]

ax.scatter(xdata, ydata, zdata, c=labels, alpha=1, cmap='tab20')
ax.set_xlabel('Self Management')
ax.set_ylabel('Social Interactions')
ax.set_zlabel('Personal Skills')

plt.title('Employees Values')
plt.show()

component_df = pd.DataFrame(transformer.components_, columns=df.iloc[:,8:].columns).T

print('Most important features for first component: (with explained variances)')
print(component_df[0].abs().sort_values(ascending=False).head(5))

print('Most important features for second component: (with explained variances)')
print(component_df[1].abs().sort_values(ascending=False).head(5))

print('Most important features for third component: (with explained variances)')
print(component_df[2].abs().sort_values(ascending=False).head(5))
import common
import em
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

df = pd.read_csv('Fire Station Database.csv')
df_values = df.iloc[:,8:].fillna(0).to_numpy()
X = df_values

mixture, post = common.init(X,4)

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
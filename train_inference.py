import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# import pickle
import common
import em

from flask import Flask
from flask import request

df = pd.read_csv('Fire Station Database.csv')
df_values = df.iloc[:,8:].fillna(0).to_numpy()
X = df_values

mixture, post = common.init(X,4)

post, cost = em.estep(X, mixture)

mixture = em.mstep(X, post, mixture)

mixture, post, cost = em.run(X, mixture, post)

X = em.fill_matrix(X, mixture)

X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)



### training part end here

### inference part starts here

labels = kmeans.labels_
print(labels)

df_result = pd.DataFrame()
df_result['color'] = labels
df_result.reset_index(inplace=True)
df_result.rename(columns={'index': 'employeeId'}, inplace=True)

df_result.to_json('result.json', orient='records')


# set Flask model and parameters
# app = Flask(__name__)

# @app.route('/predict_churn')
# def predict_churn():
#     years_in_contract = float(request.args.get('years_in_contract'))
#     age = int(request.args.get('age'))
#     num_inters = int(request.args.get('num_inters'))
#     is_male = int(request.args.get('is_male'))
#     late_on_payment = int(request.args.get('late_on_payment'))
#
#     a = {0: 'years_in_contract', 4: 'late_on_payment', 2: 'num_inters', 3: 'is_male',   1: 'age'}
#     X_temp = pd.DataFrame([years_in_contract, age, num_inters, is_male, late_on_payment]).T.rename(columns=a)
#     print(X_temp)
#     y_temp = clf_infer.predict(X_temp)
#     print(y_temp[0])
#     return str(y_temp[0])




# if __name__ == '__main__':
#     app.run(port=5000)  # for local
#     # app.run(host='0.0.0.0', port=8080)  # for AWS


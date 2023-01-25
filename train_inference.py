import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# import pickle
import common
import em

import firebase_admin
from firebase_admin import db
import json
from firebase_admin import credentials, firestore

from flask import Flask
from flask import request

def predict():
    final_dict = dict()
    for num in range(3, 7):
        mixture, post = common.init(X,4)
        post, cost = em.estep(X, mixture)
        mixture = em.mstep(X, post, mixture)
        mixture, post, cost = em.run(X, mixture, post)
        X = em.fill_matrix(X, mixture)
        X = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=num, random_state=0).fit(X)
        labels = kmeans.labels_
        result_dict = dict()
        for ind, val in enumerate(labels):
            result_dict.update({int(ind+1000): int(val)})
        final_dict.update({"cluster_num_{}".format(num): result_dict})
    return final_dict

#
#
# def write_json(labels):
#     df_result = pd.DataFrame()
#     df_result['color'] = labels
#     df_result.reset_index(inplace=True)
#     df_result.rename(columns={'index': 'employeeId'}, inplace=True)
#     df_result['employeeId'] = df_result['employeeId'] +1000
#     df_result.to_json('result.json', orient='records')


def write_firebase(final_dict):

    cd = credentials.Certificate("tmapper-a0c5d-firebase-adminsdk-uqyci-457e89c20c.json")
    firebase_admin.initialize_app(cd)

    ref = db.reference(path='/', url="https://tmapper-a0c5d-default-rtdb.europe-west1.firebasedatabase.app/")
    ref.set(final_dict)

    ## end of firebase part


# set Flask model and parameters
app = Flask(__name__)

@app.route('/service')
def service():
    pred = int(request.args.get('predict'))
    firebase = int(request.args.get('firebase'))
    if pred >0:
        final_dict = predict()
    if firebase >0:
        write_firebase(final_dict)
    return "OK"


if __name__ == '__main__':
    # app.run(port=5000)  # for local
    app.run(host='0.0.0.0', port=8080)  # for AWS


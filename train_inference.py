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
    return labels

def write_json(labels):
    df_result = pd.DataFrame()
    df_result['color'] = labels
    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index': 'employeeId'}, inplace=True)
    df_result['employeeId'] = df_result['employeeId'] +1000
    df_result.to_json('result.json', orient='records')


def write_firebase():
    ## This is FireBase part

    cd = credentials.Certificate("tmapper-a0c5d-firebase-adminsdk-uqyci-457e89c20c.json")
    firebase_admin.initialize_app(cd)

    ref = db.reference(path='/', url="https://tmapper-a0c5d-default-rtdb.europe-west1.firebasedatabase.app/")
    with open("result.json", "r") as file:
        file_contents = json.load(file)
    ref.set(file_contents)

    ## end of firebase part


# set Flask model and parameters
app = Flask(__name__)

@app.route('/service')
def service():
    pred = int(request.args.get('predict'))
    json = int(request.args.get('json'))
    firebase = int(request.args.get('firebase'))
    if pred >0:
        labels = predict()
    if json >0:
        write_json(labels)
    # if firebase >0:
    #     write_firebase()

    return "OK"




if __name__ == '__main__':
    # app.run(port=5000)  # for local
    app.run(host='0.0.0.0', port=8080)  # for AWS


import pandas as pd
import numpy as np
from flask import Flask
from flask import request
import pickle

# open pickle of the model
with open('churn_model.pkl', 'rb') as trained_model_dump:
    clf_infer = pickle.load(trained_model_dump)


# set Flask model and parameters
app = Flask(__name__)


@app.route('/predict_churn')
def predict_churn():
    years_in_contract = float(request.args.get('years_in_contract'))
    age = int(request.args.get('age'))
    num_inters = int(request.args.get('num_inters'))
    is_male = int(request.args.get('is_male'))
    late_on_payment = int(request.args.get('late_on_payment'))

    a = {0: 'years_in_contract', 4: 'late_on_payment', 2: 'num_inters', 3: 'is_male',   1: 'age'}
    X_temp = pd.DataFrame([years_in_contract, age, num_inters, is_male, late_on_payment]).T.rename(columns=a)
    print(X_temp)
    y_temp = clf_infer.predict(X_temp)
    print(y_temp[0])
    return str(y_temp[0])

# @app.route('/predict_churn_bulk', methods=['POST'])
# def predict_churn_bulk():
# work in progress


if __name__ == '__main__':
    app.run(port=8080)


import requests
import pandas as pd
import numpy as np


# X_test_client = pd.read_csv('X_test.csv')
# y_pred = np.loadtxt('preds.csv', delimiter=",")

url = 'http://ec2-3-70-239-180.eu-central-1.compute.amazonaws.com:8080'
# url = 'http://127.0.0.1:5000'
# for index in np.random.randint(0, len(X_test_client), 5):
#     row = X_test_client.iloc[index, :]
#     is_male = int(row[3])
#     num_inters = int(row[2])
#     late_on_payment = int(row[4])
#     age = int(row[1])
#     years_in_contract = row[0]
#
#     url_full = "{}/predict_churn?years_in_contract={}&late_on_payment={}&num_inters={}" \
#                "&is_male={}&age={}".format(url, years_in_contract, late_on_payment, num_inters, is_male, age)
#     response = requests.get(url_full)
#     result = str(response.content).replace("b'", "").replace("'", "")
#     print(f"For row {index+1} predicted from client is same as predicted on train {result == str(int(y_pred[index]))}")


pred = int(input('predict '))
firebase = int(input('fire '))
url_full = "{}/service?predict={}&firebase={}".format(url, pred, firebase)
response = requests.get(url_full)
result = str(response.text)
print(result)
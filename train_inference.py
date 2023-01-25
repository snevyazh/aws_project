import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask
from flask import request
import pickle


from flask import Flask
from flask import request

df = pd.read_csv('cellular_churn_greece.csv')

TARGET = 'churned'
INDEP_FEAT = ['years_in_contract', 'age', 'num_inters', 'is_male', 'late_on_payment']

X = df[INDEP_FEAT]
y = df[TARGET]

# split sets with 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


# model training and prediction
clf = RandomForestClassifier(random_state=99)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy is  {clf.score(X_test, y_test)}")


#save X test and y predicted to files
X_test.to_csv('X_test.csv', index=False)
np.savetxt('preds.csv', y_pred, delimiter = ",")

# dump model with pickle and write to file
with open('churn_model.pkl', 'wb') as model_dump:
    pickle.dump(clf, model_dump)


### training part end here

### inference part starts here

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




if __name__ == '__main__':
    app.run(port=5000)  # for local
    # app.run(host='0.0.0.0', port=8080)  # for AWS


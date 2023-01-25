import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pickle

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



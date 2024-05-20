#/usr/bin/python

import pandas as pd
import joblib

# Enter the model and data --------------------------------
rfc = joblib.load('model_rfc.pkl')  # model .pkl file

prf = pd.read_csv('/Users/igezer/ALLWISE/W3/w3_to_classify.csv')  # input data
# ---------------------------------------------------


pred_cols = list(prf.columns.values)[1:]
pred = pd.Series(rfc.predict(prf[pred_cols]))

# Get the probability of the positive class (class "real")
prob_real = pd.Series(rfc.predict_proba(prf[pred_cols])[:, 1])

# Create a new column 'class' based on the threshold (0.60) for real predictions
threshold = 0.60
pred_class = pred.copy()
pred_class[prob_real <= threshold] = 'fake'
pred_class[prob_real > threshold] = 'real'

result = pd.concat([prf['AllWISE'], pred_class, prob_real], axis=1, sort=False)
result = result.round(decimals=4)
result.columns = ['AllWISE', 'class_W3', 'Prob_W3_real']
result.to_csv('/Users/igezer/ALLWISE/W3/w3_classification_result_2.csv', header=True, index=False)

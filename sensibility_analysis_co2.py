import pandas as pd
import numpy as np
import glob

data_set = pd.read_csv("AFOLU_brazil_data_set_sensibility.csv")

data_set_all = pd.read_csv("../calibration_lac_dec/all_countries_test_CalibrationModel_class.csv")
data_set_all.query("country=='brazil'",inplace=True)

observados =  [i.split("/")[-1][:-4] for i in glob.glob("../calibration_lac_dec/observed_data/AFOLU/*.csv")]

acumula_hist = []

for i in range(312):
    acumula_hist.append(data_set_all[list(data_set.columns[:-2])+observados])

acumula_hist = pd.concat(acumula_hist,ignore_index = True)

# Declare feature vector and target variable
#X = acumula_hist
X = data_set.drop(columns=["5058","6750"])
y = data_set["5058"]

# import XGBoost
import xgboost as xgb

# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y)

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Train the XGBoost classifier

# import XGBRegressor
from xgboost import XGBRegressor
# declare parameters
params = {
            'objective':'reg:squarederror',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }
# instantiate the classifier
xgb_clf = XGBRegressor(**params)

# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

# Make predictions with XGBoost Classifier
# make predictions on test data
y_pred = xgb_clf.predict(X_test)

# check accuracy score
from sklearn.metrics import  r2_score
print('XGBoost model accuracy score: {0:0.4f}'.format(r2_score(y_test, y_pred)))

feature_importances_dict = {i:j for i,j in zip(X.columns,xgb_clf.feature_importances_)}

def insertion_sort_dict(A_dict,n):
    A = list(A_dict.values())
    llaves = list(A_dict.keys())
    i = 1
    while i < len(A):
        j = i
        while (j > 0) and (A[j-1] < A[j]):
            A[j-1],A[j] = A[j],A[j-1]
            llaves[j-1],llaves[j] = llaves[j],llaves[j-1]
            j = j - 1
        i = i +1
    return {k:v for k,v in zip(llaves[:n],A[:n])}

insertion_sort_dict(feature_importances_dict,10)


##-----------------------------------###
#     LINEAR REGRESSION
##-----------------------------------###

"""
********** Install scikit-learn **********
conda install -c anaconda scikit-learn
********** Install statsmodels  **********
conda install -c conda-forge statsmodels
"""

import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# Scale data Z-normalization


# Scale data Z-normalization
scaler = StandardScaler()
scaler.fit(X)
normalizedX = scaler.transform(X)

# Build dataframe with features normalized
data =  np.squeeze(np.hstack((normalizedX, np.array(y)[:, np.newaxis])))
pd_normalizedX = pd.DataFrame(data,columns=X.columns.to_list()+["RiskT1"])

# Run logistic regression with all the features
log_reg = smf.ols("RiskT1 ~ " + '+ '.join(X.columns), data=pd_normalizedX).fit(method='pinv')

# Build dataframe with the coeficients
coef_df = pd.DataFrame({'coef': log_reg.params.values[1:] ,
                        'err': log_reg.bse.values[1:],
                        'varname': X.columns
                       })
# Sort dataframe regarding to coefficients absolute value (ascending)
coef_df = coef_df.iloc[coef_df['coef'].abs().argsort()]

# Plot coeficients
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
coef_df.iloc[-30:].plot(x='varname', y='coef', kind='barh',
             ax=ax,legend=False , rot=0)
_ = ax.set_yticklabels(labels = coef_df.iloc[-30:].varname,fontsize=5)
plt.show()

# Sort dataframe regarding to coefficients absolute value (descending)
coef_df = coef_df.iloc[(-coef_df['coef'].abs()).argsort()].reset_index(drop=True)
coef_df.iloc[:10]["varname"]
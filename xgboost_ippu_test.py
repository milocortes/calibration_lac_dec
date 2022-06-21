from scipy.stats import qmc
import pandas as pd

# Load LAC-Decarbonization source
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd + '/lac_decarbonization/python')

import data_structures as ds
import setup_analysis as sa
import support_functions as sf
import sector_models as sm
import argparse

from model_ippu import IPPU


# Define model_ippu
model_ippu = sm.IPPU(sa.model_attributes)

df_ippu = pd.read_csv("https://raw.githubusercontent.com/egobiernoytp/lac_decarbonization/main/ref/fake_data/fake_data_ippu.csv")

df_ippu_vars = list(df_ippu.columns[1:])
# Generamos el sobol sampling para las variables gamma, nu y eta
sampler = qmc.Sobol(d=len(df_ippu_vars), scramble=False)
sample = sampler.random_base2(m=12)
l_bounds =  [0.1] * len(df_ippu_vars)
u_bounds =  [10] * len(df_ippu_vars)
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

total_df = pd.DataFrame()
df_ippu = df_ippu.iloc[:9]

for i in range(sample_scaled.shape[0]):
    print(i)
    df_ippu_test = df_ippu.copy()
    df_ippu_test = df_ippu_test[df_ippu_vars]* sample_scaled[i]
    input_pivot =  pd.concat([df_ippu[['time_period']],df_ippu_test], axis=1)
    output = model_ippu.project(input_pivot)
    partial_df = pd.DataFrame.from_dict({i:[j] for i,j in zip(df_ippu_test.columns,df_ippu_test.mean())})
    partial_df["emission_co2e_subsector_total_ippu"] = output["emission_co2e_subsector_total_ippu"].mean()
    total_df = pd.concat([total_df,partial_df])

# Declare feature vector and target variable
X = total_df.drop(columns=["emission_co2e_subsector_total_ippu"])
y = total_df["emission_co2e_subsector_total_ippu"]

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

# k-fold Cross Validation using XGBoost
from xgboost import cv

params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

# Feature importance with XGBoost
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_clf, max_num_features = 10)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

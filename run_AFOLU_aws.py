import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel,Socioeconomic,sa

import warnings

warnings.filterwarnings("ignore")

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Define target country
import sys
#target_country = str(sys.argv[1])
target_country = "brazil"
# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/emissions_targets.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_bounds/output/calib_bounds_sector.csv")

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run)).reset_index(drop = True)


"""
Cambia rangos grupos : 1,2,3,4,5

grupo    | min_35_jms    |    max_35_jms   | min_35_prop  |   max__35_pop 
 1            0.8               1.2             0.1               10.0
 2            0.1               10.0            0.1               30.0
 3            0.8               1.2             0.1               10.0 
 4            0.984252          1.023622        0.984252          1.023622
 5            0.8               1.2             0.1               10.0  
"""
calib_bounds.set_index("variable",inplace=True)
calib_bounds.loc[calib_bounds[calib_bounds["group"]==1].index,["min_35","max_35"]]= 0.1,10.0
calib_bounds.loc[calib_bounds[calib_bounds["group"]==2].index,["min_35","max_35"]]= 0.1,30.0
calib_bounds.loc[calib_bounds[calib_bounds["group"]==3].index,["min_35","max_35"]]= 0.1,10.0
calib_bounds.loc[calib_bounds[calib_bounds["group"]==5].index,["min_35","max_35"]]= 0.1,10.0
calib_bounds.reset_index(inplace=True)

calib_bounds_groups = calib_bounds.groupby("group")
indices_params = list(calib_bounds_groups.groups[0])

for i,j in calib_bounds_groups.groups.items():
    if i!=0:
        indices_params.append(j[0])

calib_targets = calib_bounds['variable'].iloc[indices_params].reset_index(drop=True)

# Define lower and upper time bounds
year_init,year_end = 2014,2019

df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and Nation=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init,year_end))
df_co2_observed_data =  df_co2_observed_data

# AFOLU FAO co2
import json
AFOLU_fao_correspondence = json.load(open("build_CO2_data_models/FAO_correspondence/AFOLU_fao_correspondence.json", "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}

"""
Calibración todos los países
"""

acumula_vectores = []
acumula_simulados_observados = []
acumula_percent_diff = []
error_percent_5058_list = []
error_percent_6750_list = []

for i in range(400):
    print(i)
    calibration = CalibrationModel(df_input_country, target_country, models_run,
                                    calib_targets, calib_bounds, t_times,
                                    df_co2_observed_data,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False)

    calibration.run_calibration("genetic_binary", population = 100, maxiter = 40)

    calib_vec_to_df = pd.DataFrame(np.array(calibration.best_vector["AFOLU"])[np.newaxis,:],columns=list(calib_targets))

    agrupa = calib_bounds.groupby("group")
    group_list = calib_bounds["group"].unique()
    total_groups = len(group_list)

    for group in group_list:
        group = int(group)
        if group == 0:
            index_var_group = calib_bounds["variable"].iloc[agrupa.groups[group]]
            calib_vec_to_df[index_var_group] =  calib_vec_to_df[index_var_group]
        else:
            index_var_group = calib_bounds["variable"].iloc[agrupa.groups[group]]
            for col in index_var_group:
                calib_vec_to_df[col] =  calib_vec_to_df[index_var_group.iloc[0]]

    error = calibration.f(calibration.best_vector["AFOLU"])
    acumula_percent_diff.append(calibration.percent_diff.to_numpy())
    
    error_percent_5058_list.append(calibration.percent_diff["5058"])
    error_percent_6750_list.append(calibration.percent_diff["6750"])

    error_percent_5058 = calibration.percent_diff["5058"].mean()
    error_percent_6750 = calibration.percent_diff["6750"].mean()

    df_error = pd.DataFrame({"error":[error]})
    df_error_percent_5058 = pd.DataFrame({"error_percent_5058":[error_percent_5058]})
    df_error_percent_6750 = pd.DataFrame({"error_percent_6750":[error_percent_6750]})

    calib_vec_to_df = pd.concat([calib_vec_to_df,df_error,df_error_percent_5058,df_error_percent_6750],axis = 1)


    acumula_vectores.append(calib_vec_to_df)

    output_data = calibration.get_output_data(calibration.best_vector["AFOLU"])

    AFOLU_fao_correspondence = json.load(open("build_CO2_data_models/FAO_correspondence/AFOLU_fao_correspondence.json", "r"))
    item_val_afolu = {}
    observed_val_afolu = {}
    for item, vars in AFOLU_fao_correspondence.items():
        if vars:
            item_val_afolu[item] = output_data[vars].sum(1).to_list()
            observed_val_afolu[item] = (df_co2_observed_data.query("Item_Code=={}".format(item)).Value/1000).to_list()

    observed_val_afolu = {k:v for k,v in observed_val_afolu.items() if len(v) > 0}

    co2_computed = pd.DataFrame(item_val_afolu).sum(axis=1)
    co2_historical = pd.DataFrame(observed_val_afolu).sum(axis=1)
    pd_computed_historical = pd.DataFrame({"country" : [target_country]*6,"iteration":[i]*6,"year":range(2014,2020),"simulado": co2_computed})
    acumula_simulados_observados.append(pd_computed_historical)



all_calib_vec_to_df = pd.concat(acumula_vectores,ignore_index=True)
all_calib_vec_to_df["nation"] = target_country
all_calib_vec_to_df["iteration"] = range(5)
df_acumula_simulados_observados = pd.concat(acumula_simulados_observados,ignore_index=True)
historicos = pd.DataFrame({"country" : [target_country]*6,"iteration":[i]*6,"year":range(2014,2020),"historico": co2_historical})

all_calib_vec_to_df.to_csv("output_calib/AFOLU/calib_vec_AFOLU_{}.csv".format(target_country),index=False)
df_acumula_simulados_observados.to_csv("output_calib/AFOLU/co2_simulados_AFOLU_{}.csv".format(target_country),index=False)
historicos.to_csv("output_calib/AFOLU/co2_historicos_AFOLU_{}.csv".format(target_country),index=False)

acumula_percent_diff_np = np.array(acumula_percent_diff)
acumula_percent_diff_np_mean = np.mean(acumula_percent_diff_np,axis=1)
df_acumula_percent_diff_np = pd.DataFrame(acumula_percent_diff_np_mean,columns = list(calibration.percent_diff.columns))
df_acumula_percent_diff_np.to_csv("output_calib/AFOLU/percent_diff_AFOLU_{}.csv".format(target_country),index=False)




X_df = []

for i in range(312):
    calib_vector = all_calib_vec_to_df.drop(columns=["error","error_percent_5058","error_percent_6750","nation"]).iloc[i].to_numpy()
    df_input_data = calibration.df_input_var.copy()
    df_input_data[calib_bounds["variable"]] = df_input_data[calib_bounds["variable"]] * calib_vector
    X_df.append(df_input_data)

X_df = pd.concat(X_df,ignore_index=True)
error_percent_5058_df = pd.concat(error_percent_5058_list,ignore_index=True)
error_percent_6750_df = pd.concat(error_percent_6750_list,ignore_index=True)

data_set_sensibility = pd.concat([X_df[calib_bounds["variable"]],error_percent_5058_df,error_percent_6750_df], axis=1,ignore_index = False)
data_set_sensibility.to_csv("AFOLU_brazil_data_set_sensibility.csv",index=False)

# Declare feature vector and target variable
X = X_df[calib_bounds["variable"]]
y = error_percent_5058_df

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

feature_importances_dict = {i:j for i,j in zip(calib_bounds["variable"],xgb_clf.feature_importances_)}

def insertion_sort_dict(A_dict):
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
    return {k:v for k,v in zip(llaves,A)}

insertion_sort_dict(feature_importances_dict)
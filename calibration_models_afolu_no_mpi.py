import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel

import itertools


import warnings
import datetime

warnings.filterwarnings("ignore")

def get_cv_index(k):
    lst_all = list(itertools.product([0, 1], repeat=7))
    lst_sub = []

    for i in lst_all:
        if sum(i)==k:
            lst_sub.append(i)

    training = []
    test = []

    for byte in lst_sub:
        index_list = [0]
        for k,i in enumerate(list(byte)):
            if i ==1:
                index_list.append(k+1)
        index_list.append(8)
        training.append(index_list)
        test.append(list(set([i for i in range(9)]) - set(index_list)))
    return training,test


# Define target country
import sys
target_country = str(sys.argv[1])

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/emissions_targets.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_bounds/output/calib_bounds_sector.csv")

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run)).reset_index(drop = True)

calib_bounds_groups = calib_bounds.groupby("group")
indices_params = list(calib_bounds_groups.groups[0])

for i,j in calib_bounds_groups.groups.items():
    if i!=0:
        indices_params.append(j[0])

calib_targets = calib_bounds['variable'].iloc[indices_params].reset_index(drop=True)

# Define lower and upper time bounds
year_init,year_end = 2011,2019

df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and Nation=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init,year_end))
df_co2_observed_data =  df_co2_observed_data

# AFOLU FAO co2
import json
AFOLU_fao_correspondence = json.load(open("build_CO2_data_models/FAO_correspondence/AFOLU_fao_correspondence.json", "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}

# Define dataframe to save results
pd_output_all = pd.DataFrame()

# Define list by MPI execution
sendbuf_test = []
sendbuf_training = []
sendbuf_cv_index = []

# Total process number
proc_n = 8

print("TIME START CV CALIBRATION {}\n{}".format(target_country,datetime.datetime.now().time()))
training,test = get_cv_index(4)
training,test = np.array(training) , np.array(test)
sendbuf_test = np.array_split(test,proc_n)
sendbuf_training = np.array_split(training,proc_n)
sendbuf_cv_index = np.array_split([i for i in range(training.shape[0])],proc_n)

for proc_id,(iteracion,training,test) in enumerate(zip(sendbuf_cv_index,sendbuf_training,sendbuf_test)): 
    for k,cv,cv_test in zip(iteracion,training,test):
        # Optimize the function
        import time
        start_time = time.time()

        calibration = CalibrationModel(df_input_country, target_country, models_run,
                                    calib_targets, calib_bounds, t_times,
                                    df_co2_observed_data,AFOLU_fao_correspondence,cv_training = cv , 
                                    cv_test = cv_test, cv_run = k, id_mpi = proc_id,
                                    cv_calibration = True)

        calibration.run_calibration("genetic_binary", population = 10, maxiter = 10)

        mse_test = calibration.get_mse_test(calibration.best_vector[models_run])
        print(" Cross Validation: {} on Node {}. MSE Test : {}".format(k,proc_id,mse_test))

        pd_output = pd.DataFrame.from_dict({target:[value] for target,value in zip(calib_targets,calibration.best_vector[models_run])})
        pd_output["nation"] = target_country
        pd_output["MSE_training"] = calibration.fitness_values[models_run][-1]
        pd_output["MSE_test"] = mse_test
        pd_output_all = pd.concat([pd_output_all, pd_output])

    target_country = target_country.replace(" ","_")
    target_country = target_country.lower()
    pd_output_all.to_csv("output_calib/{}/mpi_output/{}_cv_{}_mpi_{}.csv".format(models_run,models_run,target_country,proc_id), index = None, encoding = "UTF-8")

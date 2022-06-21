import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

import itertools
from mpi4py import MPI


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

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Define target country
target_country = str(sys.argv[1])


# Set model to run
models_run = "CircularEconomy"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_CO2_data_models/output/df_calib_targets_models.csv")
calib_targets = df_calib_targets.query("model == '{}'".format(models_run))["calib_targets"]
# Load observed data
observed_by_sector = pd.read_csv("observed_data/summary_observed_by_sector.csv")
# Load all variables of sector
all_by_sector = pd.read_csv("observed_data/all_variables_CircularEconomy.csv")

list_validate_category = list(all_by_sector["var"][["frac_wali_ww_domestic_rural_treatment_path" in i or "frac_wali_ww_domestic_urban_treatment_path" in i or "frac_wali_ww_industrial_treatment_path" in i for i in all_by_sector["var"]]])

# Add to calib_targets the difference between all_by_sector and  calib_targets + observed_by_sector
excluye = list(set(list(calib_targets)+list(observed_by_sector["observed_var"]) + list_validate_category))
calib_05_150 = list(set(all_by_sector["var"]).difference(set(excluye)))

# calibration bounds
calib_bounds = pd.read_csv("bounds/model_input_variables_ce_demo.csv")
calib_bounds = calib_bounds[["variable","min_35","max_35"]]
calib_bounds = calib_bounds.query("min_35 != 1.00 and max_35 !=1.00")
calib_bounds = calib_bounds[[False if i in ['gasrf_waso_biogas','oxf_waso_average_landfilled','elasticity_protein_in_diet_to_gdppc','frac_waso_compost_methane_flared','gasrf_waso_landfill_to_ch4']  else True for i in calib_bounds["variable"]]]

calib_bounds = pd.concat([calib_bounds,pd.DataFrame.from_dict({'variable':calib_05_150 , 'min_35':[0.5]*len(calib_05_150),'max_35':[1.5]*len(calib_05_150)})])
calib_targets = calib_bounds["variable"]
# Define lower and upper time bounds
year_init,year_end = 2006,2014

df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))

# Define dataframe to save results
pd_output_all = pd.DataFrame()

# Define list by MPI execution
sendbuf_test = []
sendbuf_training = []
sendbuf_cv_index = []

# The process 0 will be de master. His function is create de initial matriz for Cross-Validations runs and distribute it each the process
root=0
comm = MPI.COMM_WORLD


print("TIME START CV CALIBRATION {}\n{}".format(target_country,datetime.datetime.now().time()))
if comm.rank==0:
    training,test = get_cv_index(4)
    training,test = np.array(training) , np.array(test)
    sendbuf_test = np.array_split(test,comm.size)
    sendbuf_training = np.array_split(training,comm.size)
    sendbuf_cv_index = np.array_split([i for i in range(training.shape[0])],comm.size)

test_chunk = comm.scatter(sendbuf_test,root)
training_chunk = comm.scatter(sendbuf_training,root)
index_chunk = comm.scatter(sendbuf_cv_index,root)

for k,cv,cv_test in zip(index_chunk,training_chunk,test_chunk):
    # Optimize the function
    import time
    start_time = time.time()

    calibration = CalibrationModel(df_input_country, target_country, models_run,
                                calib_targets, calib_bounds, t_times,
                                df_co2_observed_data,cv_training = cv , cv_test = cv_test,cv_calibration = True, cv_run = k, id_mpi = comm.rank)
    calibration.run_calibration("genetic_binary", population = 60, maxiter = 10)

    mse_test = calibration.get_mse_test(calibration.best_vector["CircularEconomy"])
    print(" Cross Validation: {} on Node {}. MSE Test : {}".format(k,comm.rank,mse_test))

    pd_output = pd.DataFrame.from_dict({target:[value] for target,value in zip(calib_targets,calibration.best_vector["CircularEconomy"])})
    pd_output["nation"] = target_country
    pd_output["MSE_training"] = calibration.fitness_values
    pd_output["MSE_test"] = mse_test
    pd_output_all = pd.concat([pd_output_all, pd_output])

target_country = target_country.replace(" ","_")
target_country = target_country.lower()
pd_output_all.to_csv("output_calib/{}/mpi_output/{}_cv_{}_mpi_{}.csv".format(models_run,models_run,target_country,comm.rank), index = None, encoding = "UTF-8")

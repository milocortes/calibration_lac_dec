import os

os.environ['LAC_PATH'] = '/home/milo/Documents/julia/lac_decarbonization'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sisepuede_calibration.calibration_lac import CalibrationModel

data_path = "/home/milo/Documents/egap/SISEPUEDE/packaging_projects/github_projects/sisepuede_calibration_dev/unit_testing/data_test"

df_input_all_countries = pd.read_csv( os.path.join(data_path, "real_data_2022_10_04.csv"))

# Define target country
target_country = "brazil"

# Set model to run
models_run = "IPPU"

# Define lower and upper time bounds
year_init,year_end = 2014,2019

# Load observed CO2 data
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "ghg_LAC_circular_ippu.csv") )


# Load calib targets by model to run
df_calib_targets =  pd.read_csv( os.path.join(data_path, "calib_bounds_sector.csv") )

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_bounds = calib_bounds.query("variable != 'gdp_mmm_usd'")

calib_targets_ippu = calib_bounds['variable']


df_co2_observed_data = df_co2_observed_data.query("model == '{}' and Nation=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init,year_end))

df_co2_observed_data.rename(columns = {"Value" : "value"}, inplace = True)

df_input_country = df_input_all_countries.query("Nation =='{}' and (Year>={} and Year<={})".format(target_country,year_init,year_end)).reset_index().drop(columns=["index"])
df_input_country_all_time_period = df_input_all_countries.query("Nation =='{}'".format(target_country)).reset_index().drop(columns=["index"])

# Instance of CalibrationModel
calibration_ippu = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets_ippu, calib_bounds,df_input_country_all_time_period,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4)
# Test function evaluation
X = [np.mean((calibration_ippu.df_calib_bounds.loc[calibration_ippu.df_calib_bounds["variable"] == i, "min_35"].item(),calibration_ippu.df_calib_bounds.loc[calibration_ippu.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration_ippu.calib_targets["IPPU"]]

calibration_ippu.f(X)

# Setup and run calibration with PSO

parameters_algo = {"alpha" : 0.5, "beta" : 0.8}

calibration_ippu.run_calibration("pso", population = 20, maxiter = 10, param_algo = parameters_algo)

# Check fitness
plt.plot(calibration_ippu.fitness_values["IPPU"])
plt.show()

calibration_vector_IPPU = calibration_ippu.best_vector["IPPU"]

output_data = calibration_ippu.get_output_data(calibration_vector_IPPU)

co2_computed = output_data[calibration_ippu.var_co2_emissions_by_sector["IPPU"]].sum(axis=1)
plt.plot([i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(target_country)
plt.legend()
plt.show()

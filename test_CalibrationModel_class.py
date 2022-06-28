import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import CalibrationModel

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Define target country
target_country = "mexico"

# Set model to run
models_run = "CircularEconomy"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_bounds/output/calib_bounds_sector.csv")

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets = calib_bounds['variable']

# Define lower and upper time bounds
year_init,year_end = 2006,2014

df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))

calibration = CalibrationModel(df_input_country, target_country, models_run,
                                calib_targets, calib_bounds, t_times,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8] , cv_test = [1,3,5,7])

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration.calib_targets["CircularEconomy"]]
calibration.f(X)
calibration.run_calibration("genetic_binary", population = 60, maxiter = 10)
plt.plot(calibration.fitness_values["CircularEconomy"])
plt.show()


output_data = calibration.get_output_data(calibration.best_vector["CircularEconomy"])
co2_computed = output_data[calibration.var_co2_emissions_by_sector["CircularEconomy"]].sum(axis=1)
plt.plot([i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title("mexico")
plt.legend()
plt.show()

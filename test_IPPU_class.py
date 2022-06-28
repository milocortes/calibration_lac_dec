import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Define target country
target_country = "brazil"

# Set model to run
models_run = "IPPU"

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
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8] , cv_test = [1,3,5,7], cv_calibration = True)

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration.calib_targets["IPPU"]]

calibration.f(X)
calibration.run_calibration("genetic_binary", population = 60, maxiter = 10)
plt.plot(calibration.fitness_values["IPPU"])
plt.show()


output_data = calibration.get_output_data(calibration.best_vector["IPPU"])

co2_computed = output_data[calibration.var_co2_emissions_by_sector["IPPU"]].sum(axis=1)
plt.plot([i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(target_country)
plt.legend()
plt.show()

calibration.get_output_data(X)

## Agregamos los vectores de calibración de waste
calib_waste = pd.read_csv("/home/milo/Documents/egtp/LAC-dec/calibration/output_calib/CircularEconomy/calib_vector_circular_economy_all_countries.csv")

calib_targets_waste = list(calib_waste.columns[:-2])

models_run = "CircularEconomy"
calibration_waste = CalibrationModel(df_input_country, target_country, models_run,
                                calib_targets_waste, calib_bounds, t_times,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8] , cv_test = [1,2,3,4,5,6,7,8])

df_output_data = calibration_waste.get_output_data(list(calib_waste.iloc[0][:-2]))


df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_country,
                df_output_data,
                sm.IPPU(sa.model_attributes).integration_variables
            )

model_ippu = sm.IPPU(sa.model_attributes)
model_ippu.project(df_input_data)

## Agregamos los vectores de calibración de CircularEconomy en clase Calubration
calib_circecon = pd.read_csv("/home/milo/Documents/egtp/LAC-dec/calibration/output_calib/CircularEconomy/calib_vector_circular_economy_all_countries.csv")
calib_circecon_calib_targets = list(calib_circecon.columns[:-2])
target_country_index = list(calib_circecon["country"]).index(target_country)
calib_circecon_calib_vector = list(calib_circecon.iloc[target_country_index][:-2])

calibration = CalibrationModel(df_input_country, target_country, models_run,
                                calib_targets, calib_bounds, t_times,
                                df_co2_observed_data,cv_training = [0,2,4,6,8] ,
                                cv_test = [1,3,5,7],
                                cv_calibration = True,
                                downstream = True)
                                
calibration.set_calib_targets('CircularEconomy',calib_circecon_calib_targets)
calibration.set_best_vector('CircularEconomy',calib_circecon_calib_vector)

calibration.run_calibration("genetic_binary", population = 60, maxiter = 10)




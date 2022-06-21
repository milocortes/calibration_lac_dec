import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Set model to run
models_run = "IPPU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")
df_calib_targets =  pd.read_csv("model_input_variables_ip_demo.csv")
df_calib_targets = df_calib_targets.drop_duplicates()

# Load fake_data_ippu
df_ippu = pd.read_csv("https://raw.githubusercontent.com/egobiernoytp/lac_decarbonization/main/ref/fake_data/fake_data_ippu.csv")

coincide_target_fake_ippu = list(set(df_ippu.columns).intersection(set(df_calib_targets["variable"])))

df_calib_targets = df_calib_targets[[True if i in coincide_target_fake_ippu else False for i in df_calib_targets["variable"]]]

# Load observed data
observed_by_sector = ["prodinit_ippu_chemicals_tonne",
"prodinit_ippu_plastic_tonne",
"prodinit_ippu_metals_tonne",
"prodinit_ippu_electronics_tonne",
"prodinit_ippu_textiles_tonne",
"prodinit_ippu_paper_tonne",
"prodinit_ippu_lime_and_carbonite_tonne"]


# calibration bounds
calib_bounds = df_calib_targets[["variable","min_35","max_35"]]
calib_targets = calib_bounds["variable"]
calib_targets = list(set(calib_targets).difference(observed_by_sector))
calib_bounds = calib_bounds.query("min_35 != 1.00 and max_35 !=1.00")
calib_bounds = calib_bounds[[not i if i in observed_by_sector else True for i in calib_bounds["variable"]]]
calib_bounds = calib_bounds[[not i for i in calib_bounds['min_35'].isna() |  calib_bounds['max_35'].isna()]]

calib_targets_05_1_50 = list(set(calib_targets).difference(calib_bounds["variable"]))
calib_bounds_05_1_50 = pd.DataFrame.from_dict({'variable': calib_targets_05_1_50, 'min_35':[0.5]*len(calib_targets_05_1_50),'max_35':[1.5]*len(calib_targets_05_1_50)})

calib_targets_05_1_50 += ["gdp_mmm_usd"]
calib_bounds_05_1_50 = pd.concat([calib_bounds_05_1_50, pd.DataFrame.from_dict({'variable':['gdp_mmm_usd'],'min_35':[0.005],'max_35':[2]})])

calib_bounds = pd.concat([calib_bounds,calib_bounds_05_1_50])

calib_targets = calib_bounds["variable"]

# Define lower and upper time bounds
year_init,year_end = 2006,2014

t_times = range(year_init, year_end+1)

# Add variable that change

countries_list = list(set(df_co2_observed_data.query("model == 'IPPU'")["country"]))
countries_list.sort()

target_country = "brazil"

df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))

calibration = CalibrationModel(df_input_country, target_country, models_run,
                                calib_targets, calib_bounds, t_times,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8] , cv_test = [1,3,5,7])

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration.calib_targets]

calibration.f(X)
calibration.run_calibration("genetic_binary", population = 60, maxiter = 20)
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
calib_waste = pd.read_csv("/home/milo/Documents/egap/test/pruebas/calibration/calibration/output_calib/CircularEconomy/calib_vector_circular_economy_all_countries.csv")

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

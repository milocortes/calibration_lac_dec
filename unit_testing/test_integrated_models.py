import os

os.environ['LAC_PATH'] = '/home/milo/Documents/julia/lac_decarbonization'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sisepuede_calibration.calibration_lac import CalibrationModel

data_path = "https://raw.githubusercontent.com/milocortes/calibration_lac_dec/main/unit_testing/data_test"

#df_input_all_countries = pd.read_csv( os.path.join(data_path, "real_data_2023_01_11.csv"))
df_input_all_countries = pd.read_csv( os.path.join(data_path, "real_data_2023_01_11_energy.csv"))

# Define target country
target_country = "chile"

"""

#### RUN AFOLU MODEL

"""

# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "emissions_targets.csv") )

# Load calib targets by model to run
df_calib_targets =  pd.read_csv( os.path.join(data_path, "calib_bounds_sector.csv") )

remueve_calib = ['qty_soil_organic_c_stock_dry_climate_tonne_per_ha',
 'qty_soil_organic_c_stock_temperate_crop_grass_tonne_per_ha',
 'qty_soil_organic_c_stock_temperate_forest_nutrient_poor_tonne_per_ha',
 'qty_soil_organic_c_stock_temperate_forest_nutrient_rich_tonne_per_ha',
 'qty_soil_organic_c_stock_tropical_crop_grass_tonne_per_ha',
 'qty_soil_organic_c_stock_tropical_forest_tonne_per_ha',
 'qty_soil_organic_c_stock_wet_climate_tonne_per_ha',
 'scalar_lvst_carrying_capacity','frac_soil_soc_loss_in_cropland']

df_calib_targets = df_calib_targets[~df_calib_targets.variable.isin(remueve_calib)]
calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run)).reset_index(drop = True)
#calib_bounds = calib_bounds.query("variable !='pij_calib'")

calib_bounds_groups = calib_bounds.groupby("group")
indices_params = list(calib_bounds_groups.groups[0])

for i,j in calib_bounds_groups.groups.items():
    if i!=0:
        indices_params.append(j[0])

calib_targets = calib_bounds['variable'].iloc[indices_params].reset_index(drop=True)
#calib_targets = calib_targets.append(pd.Series("pij_calib"),ignore_index=True)


# Define lower and upper time bounds
year_init,year_end = 2014,2019

df_input_country = df_input_all_countries.query("Nation =='{}' and (Year>={} and Year<={})".format(target_country,year_init,year_end)).reset_index().drop(columns=["index"])
df_input_country["time_period"] = list(range(1+(year_end-year_init)))


t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and Nation=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init,year_end))
df_co2_observed_data =  df_co2_observed_data

df_input_country_all_time_period = df_input_all_countries.query("Nation =='{}'".format(target_country)).reset_index().drop(columns=["index"])

# AFOLU FAO co2
import json
AFOLU_fao_correspondence = json.load(open("/home/milo/Documents/egap/SISEPUEDE/packaging_projects/minimal/AFOLU_fao_correspondence.json", "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}

calibration = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets, df_calib_targets,df_input_country_all_time_period,
                                df_co2_observed_data,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4)

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item()))  for i in calibration.calib_targets["AFOLU"]]

calibration.f(X)

param_algo = {"alpha" : 0.5, "beta" : 0.8}

calibration.run_calibration("pso", population = 20, maxiter = 10, param_algo = param_algo)

plt.plot(calibration.fitness_values["AFOLU"])
plt.show()

calibration_vector_AFOLU = calibration.best_vector["AFOLU"]

output_data = calibration.get_output_data(calibration_vector_AFOLU, print_sector_model = True)
#calibration.build_bar_plot_afolu(calibration.best_vector["AFOLU"], show = True)

item_val_afolu = {}
observed_val_afolu = {}
for item, vars in AFOLU_fao_correspondence.items():
    if vars:
        item_val_afolu[item] = output_data[vars].sum(1).to_list()
        observed_val_afolu[item] = (df_co2_observed_data.query("Item_Code=={}".format(item)).Value/1000).to_list()

observed_val_afolu = {k:v for k,v in observed_val_afolu.items() if len(v) > 0}

co2_computed = pd.DataFrame(item_val_afolu).sum(axis=1)
co2_historical = pd.DataFrame(observed_val_afolu).sum(axis=1)

plt.plot(co2_historical,label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
plt.show()


"""

#### RUN CircularEconomy MODEL

"""

models_run = "CircularEconomy"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "ghg_LAC_circular_ippu.csv") )


calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets_circular_economy = calib_bounds['variable']

df_co2_observed_data = df_co2_observed_data.query("model == '{}' and Nation=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init,year_end))

df_co2_observed_data.rename(columns = {"Value" : "value"}, inplace = True)


# Instance of CalibrationModel
calibration_circ_ec = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets_circular_economy, calib_bounds,df_input_country_all_time_period,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)
# Test function evaluation
X = [np.mean((calibration_circ_ec.df_calib_bounds.loc[calibration_circ_ec.df_calib_bounds["variable"] == i, "min_35"].item(),calibration_circ_ec.df_calib_bounds.loc[calibration_circ_ec.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration_circ_ec.calib_targets["CircularEconomy"]]
calibration_circ_ec.set_best_vector("AFOLU",calibration_vector_AFOLU)
calibration_circ_ec.f(X)


# Setup and run calibration with PSO

parameters_algo = {"alpha" : 0.5, "beta" : 0.8}

calibration_circ_ec.run_calibration("pso", population = 20, maxiter = 10, param_algo = parameters_algo)

# Check fitness
plt.plot(calibration_circ_ec.fitness_values["CircularEconomy"])
plt.show()


# Check performance
calibration_vector_CircularEconomy = calibration_circ_ec.best_vector["CircularEconomy"]

output_data = calibration_circ_ec.get_output_data(calibration_vector_CircularEconomy, print_sector_model = True)

co2_computed = output_data[calibration_circ_ec.var_co2_emissions_by_sector["CircularEconomy"]].sum(axis=1)
plt.plot(range(year_init,year_end+1),[i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(range(year_init,year_end+1),co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
plt.show()


"""

#### RUN IPPU MODEL

"""

# Set model to run
models_run = "IPPU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "ghg_LAC_circular_ippu.csv") )


calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets_ippu = calib_bounds['variable']


df_co2_observed_data = df_co2_observed_data.query("model == '{}' and Nation=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init,year_end))

df_co2_observed_data.rename(columns = {"Value" : "value"}, inplace = True)


# Instance of CalibrationModel
calibration_ippu = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets_ippu, calib_bounds,df_input_country_all_time_period,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)
# Test function evaluation
X = [np.mean((calibration_ippu.df_calib_bounds.loc[calibration_ippu.df_calib_bounds["variable"] == i, "min_35"].item(),calibration_ippu.df_calib_bounds.loc[calibration_ippu.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration_ippu.calib_targets["IPPU"]]

calibration_ippu.set_best_vector("AFOLU",calibration_vector_AFOLU)
calibration_ippu.set_best_vector("CircularEconomy",calibration_vector_CircularEconomy)
calibration_ippu.set_calib_targets("CircularEconomy", calib_targets_circular_economy)
calibration_ippu.f(X)

# Setup and run calibration with PSO

parameters_algo = {"alpha" : 0.5, "beta" : 0.8}

calibration_ippu.run_calibration("pso", population = 20, maxiter = 10, param_algo = parameters_algo)

# Check fitness
plt.plot(calibration_ippu.fitness_values["IPPU"])
plt.show()

calibration_vector_IPPU = calibration_ippu.best_vector["IPPU"]

output_data = calibration_ippu.get_output_data(calibration_vector_IPPU, print_sector_model = True)

co2_computed = output_data[calibration_ippu.var_co2_emissions_by_sector["IPPU"]].sum(axis=1)
plt.plot([i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
plt.show()

output_data_ippu = calibration_ippu.get_output_data(calibration_vector_IPPU, print_sector_model = True)

"""

#### RUN NonElectricEnergy MODEL

"""

# Set model to run
models_run = "NonElectricEnergy"


# Instance of CalibrationModel
calibration_NoEenergy = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets_ippu, calib_bounds,df_input_country_all_time_period,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)

calibration_NoEenergy.set_best_vector("AFOLU",calibration_vector_AFOLU)

calibration_NoEenergy.set_best_vector("CircularEconomy",calibration_vector_CircularEconomy)
calibration_NoEenergy.set_calib_targets("CircularEconomy", calib_targets_circular_economy)

calibration_NoEenergy.set_best_vector("IPPU",calibration_vector_IPPU)
calibration_NoEenergy.set_calib_targets("IPPU", calib_targets_ippu)

output_data_NonElectricEnergy = calibration_NoEenergy.get_output_data([1], print_sector_model = True)

"""

#### RUN ElectricEnergy MODEL

"""

# Set model to run
models_run = "ElectricEnergy"


# Instance of CalibrationModel
calibration_Eenergy = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets_ippu, calib_bounds,df_input_country_all_time_period,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)

calibration_Eenergy.set_best_vector("AFOLU",calibration_vector_AFOLU)

calibration_Eenergy.set_best_vector("CircularEconomy",calibration_vector_CircularEconomy)
calibration_Eenergy.set_calib_targets("CircularEconomy", calib_targets_circular_economy)

calibration_Eenergy.set_best_vector("IPPU",calibration_vector_IPPU)
calibration_Eenergy.set_calib_targets("IPPU", calib_targets_ippu)

output_data_ElectricEnergy = calibration_Eenergy.get_output_data([1], print_sector_model = True)

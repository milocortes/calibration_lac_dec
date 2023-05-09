import os

os.environ['LAC_PATH'] = '/home/milo/Documents/egap/calibration/lac_decarbonization'

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sisepuede_calibration.sisepuede_calibration import CalibrationModel, SectorIO, sa, sf, sm


# Set directories
#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.getcwd()
data_path = os.path.abspath(os.path.join(dir_path,"..","data","inputs_sisepuede" ))
save_data_path = os.path.abspath(os.path.join(dir_path,"..","output" ))

# Load input data 
df_input_all_countries = pd.read_csv( os.path.join(data_path, "real_data_20230509.csv"))

# Define target country
#target_country = sys.argv[1]
target_country = "MEX"

#### Load calib targets by model to run
## Calib bounds
df_calib_targets =  pd.read_csv( os.path.join(data_path, "calib_bounds_sector_energy_subsectors.csv") )
df_calib_targets = df_calib_targets.query("variable != 'gdp_mmm_usd'")
### Load calib targets
calib_targets_all_sectors =  pd.read_csv( os.path.join(data_path, "calib_targets_all_sectors_energy_subsectors.csv") )

# Load observed CO2 data
df_co2_observed_data = {"AFOLU" : pd.read_csv(os.path.join(data_path, "emissions_targets_promedios_iso_code3.csv")),
                        "CircularEconomy": pd.read_csv(os.path.join(data_path, "ghg_LAC_circular_ce_iso_code3.csv")),
                        "IPPU" : pd.read_csv(os.path.join(data_path, "ghg_LAC_circular_ippu_iso_code3.csv")),
                        "NonElectricEnergy" : pd.read_csv(os.path.join(data_path, "ghg_NonElectricEnergy_iso_code3.csv")),
                        "ElectricEnergy" : pd.read_csv(os.path.join(data_path, "ghg_ElectricEnergy_iso_code3.csv")),
                        "Fugitive" : pd.read_csv(os.path.join(data_path, "ghg_Fugitive_iso_code3.csv"))
                        }


# Define lower and upper time bounds
year_init,year_end = 0,5

df_input_all_countries["time_period"] = df_input_all_countries["Year"] - 2015


# Subset input data
df_input_country = df_input_all_countries.query("iso_code3 =='{}' and (time_period>={} and time_period<={})".format(target_country,year_init,year_end)).reset_index().drop(columns=["index"])

df_input_country_all_time_period = df_input_all_countries.query("iso_code3 =='{}'".format(target_country)).reset_index().drop(columns=["index"])

##### Load Crosswalks
import json

### AFOLU FAO crosswalk co2 - SISEPUEDE
AFOLU_fao_correspondence = json.load(open(os.path.join(data_path,"AFOLU_fao_correspondence.json"), "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}

## Load Energy crosswalk
energy_correspondence = json.load(open(os.path.join(data_path, "energy_subsector_items.json") , "r"))


var_co2_emissions_by_sector = {'CircularEconomy' : ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"],
                                            'IPPU': ['emission_co2e_subsector_total_ippu'],
                                            'AFOLU' : AFOLU_fao_correspondence,
                                            'AllEnergy' : energy_correspondence}


"""

#### RUN IPPU CALIBRATION

"""

# Set model to run
models_run = "IPPU"

# Instance of CalibrationModel
calibration = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                    df_calib_targets, df_input_country_all_time_period,
                                    df_co2_observed_data, var_co2_emissions_by_sector, precition=4)

# Test function evaluation with a random calibration vector
X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration.calib_targets["IPPU"]]

# Evaluate random calibration vector in the function to minimize
calibration.f(X)

# Setup optimization algorithm parameters and run calibration with PSO  
parameters_algo = {"alpha" : 0.5, "beta" : 0.8}
calibration.run_calibration("pso", population = 10, maxiter = 10, param_algo = parameters_algo)

# Check fitness
#plt.plot(calibration_ippu.fitness_values["IPPU"])
#plt.show()

calibration_vector_IPPU = calibration.best_vector["IPPU"]

with open(os.path.join(save_data_path, "calib_vectors", f'calibration_vector_IPPU_{target_country}.pickle'), 'wb') as f:
    pickle.dump(calibration_vector_IPPU, f)

output_data = calibration.get_output_data(calibration_vector_IPPU, print_sector_model = True)

co2_computed = output_data[calibration.var_co2_emissions_by_sector["IPPU"]].sum(axis=1)
df_co2_observed_data_ippu = df_co2_observed_data[models_run].query(f"model == '{models_run}' and iso_code3=='{target_country}' and (Year >= {year_init+2014} and Year <= {year_end+2014} )").reset_index(drop = True).copy()
plt.plot([i/1000 for i in df_co2_observed_data_ippu.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
#plt.savefig(os.path.join(save_data_path, f"{models_run}_{target_country}.png"))
#plt.close()
plt.show()

#### MERGE INPUT AND OUTPUT DATA
df_input_country_all_time_period = calibration.get_calibrated_data(calibration_vector_IPPU)
df_input_country = df_input_country_all_time_period.iloc[:6].copy()

"""

#### RUN AFOLU CALIBRATION

"""

# Set model to run
models_run = "AFOLU"

calib_targets_afolu = calib_targets_all_sectors.query(f"sector =='{models_run}'")["calib_targets"].reset_index(drop = True).to_list()[:-1]

calibration.update_model(models_run, df_input_country, df_input_country_all_time_period, calib_targets_afolu)

# Test function evaluation with a random calibration vector
X = [1] * len(calibration.calib_targets[models_run])

# Evaluate random calibration vector in the function to minimize
calibration.f(X)

# Setup optimization algorithm parameters and run calibration with PSO  
param_algo = {"alpha" : 0.5, "beta" : 0.8}
calibration.run_calibration("pso", population = 10, maxiter = 10, param_algo = param_algo)

#plt.plot(calibration.fitness_values["AFOLU"])
#plt.show()

# Get calibration vector
calibration_vector_AFOLU = calibration.best_vector["AFOLU"]

# Save calibration vector
with open(os.path.join(save_data_path, "calib_vectors", f'calibration_vector_AFOLU_{target_country}.pickle'), 'wb') as f:
    pickle.dump(calibration_vector_AFOLU, f)

# Plot historical vs simulated data
output_data = calibration.get_output_data(calibration_vector_AFOLU, print_sector_model = True)

#calibration.build_bar_plot_afolu(calibration.best_vector["AFOLU"], show = True)
df_co2_observed_data_afolu = df_co2_observed_data[models_run].query(f"model == '{models_run}' and iso_code3=='{target_country}' and (Year >= {year_init+2014} and Year <= {year_end+2014} )").reset_index(drop = True).copy()

item_val_afolu = {}
observed_val_afolu = {}
for item, vars in AFOLU_fao_correspondence.items():
    if vars:
        item_val_afolu[item] = output_data[vars].sum(1).to_list()
        observed_val_afolu[item] = (df_co2_observed_data_afolu.query("Item_Code=={}".format(item)).Value/1000).to_list()

observed_val_afolu = {k:v for k,v in observed_val_afolu.items() if len(v) > 0}

co2_computed = pd.DataFrame(item_val_afolu).sum(axis=1)
co2_historical = pd.DataFrame(observed_val_afolu).sum(axis=1)

plt.plot(co2_historical,label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
#plt.savefig(os.path.join(save_data_path, f"{models_run}_{target_country}.png"))
#plt.close()
plt.show()

#### MERGE INPUT AND OUTPUT DATA
df_input_country_all_time_period = calibration.get_calibrated_data(calibration_vector_AFOLU)
df_input_country = df_input_country_all_time_period.iloc[:6].copy()

"""

#### RUN CircularEconomy CALIBRATION

"""

# Set model to run
models_run = "CircularEconomy"

calibration.update_model(models_run, df_input_country, df_input_country_all_time_period)

# Test function evaluation with a random calibration vector
X = [1] * len(calibration.calib_targets[models_run])

# Evaluate random calibration vector in the function to minimize
calibration.f(X)

# Setup optimization algorithm parameters and run calibration with PSO  
param_algo = {"alpha" : 0.5, "beta" : 0.8}
calibration.run_calibration("pso", population = 10, maxiter = 10, param_algo = param_algo)

# Check performance
calibration_vector_CircularEconomy = calibration.best_vector["CircularEconomy"]


with open(os.path.join(save_data_path, "calib_vectors", f'calibration_vector_CircularEconomy_{target_country}.pickle'), 'wb') as f:
    pickle.dump(calibration_vector_CircularEconomy, f)

output_data = calibration.get_output_data(calibration_vector_CircularEconomy, print_sector_model = True)

co2_computed = output_data[calibration.var_co2_emissions_by_sector["CircularEconomy"]].sum(axis=1)
df_co2_observed_data_circular_economy = df_co2_observed_data[models_run].query(f"model == '{models_run}' and iso_code3=='{target_country}' and (Year >= {year_init+2014} and Year <= {year_end+2014} )").reset_index(drop = True).copy()

plt.plot(range(year_init,year_end+1),[i/1000 for i in df_co2_observed_data_circular_economy.value],label="historico")
plt.plot(range(year_init,year_end+1),co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
#plt.savefig(os.path.join(save_data_path, f"{models_run}_{target_country}.png"))
#plt.close()
plt.show()

#### MERGE INPUT AND OUTPUT DATA
df_input_country_all_time_period = calibration.get_calibrated_data(calibration_vector_CircularEconomy)
df_input_country = df_input_country_all_time_period.iloc[:6].copy()




"""

#### RUN NonElectricEnergy CALIBRATION

"""


models_run = "NonElectricEnergy"

# CircularEconomy calibration targets
calib_targets_NonElectricEnergy = calib_targets_all_sectors.query(f"sector =='{models_run}'")["calib_targets"].reset_index(drop = True)

# CircularEconomy CO2 observed data
df_co2_observed_data_NonElectricEnergy = df_co2_observed_data[models_run].query("model == '{}' and iso_code3=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init+2014,year_end+2014))

# Instance of CalibrationModel
calibration_circ_NonElectricEnergy = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets_NonElectricEnergy, df_calib_targets, df_input_country_all_time_period,
                                df_co2_observed_data_NonElectricEnergy, cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4)

# Test function evaluation with a random calibration vector
X = [np.mean((calibration_circ_ec.df_calib_bounds["CircularEconomy"].loc[calibration_circ_ec.df_calib_bounds["CircularEconomy"]["variable"] == i, "min_35"].item(),calibration_circ_ec.df_calib_bounds["CircularEconomy"].loc[calibration_circ_ec.df_calib_bounds["CircularEconomy"]["variable"] == i, "max_35"].item() +0.01))  for i in calibration_circ_ec.calib_targets["CircularEconomy"]]

# Evaluate random calibration vector in the function to minimize
calibration_circ_ec.f(X)


# Setup optimization algorithm parameters and run calibration with PSO  
parameters_algo = {"alpha" : 0.5, "beta" : 0.8}
calibration_circ_ec.run_calibration("pso", population = 10, maxiter = 10, param_algo = parameters_algo)

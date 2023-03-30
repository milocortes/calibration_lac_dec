import os

os.environ['LAC_PATH'] = '/home/milo/Documents/egap/calibration/lac_decarbonization'

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sisepuede_calibration.calibration_lac import CalibrationModel


# Set directories
#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.getcwd()
data_path = os.path.abspath(os.path.join(dir_path,"..","data","inputs_sisepuede" ))
save_data_path = os.path.abspath(os.path.join(dir_path,"..","output" ))

# Correspondence iso code 3 - SISEPUEDE
iso3_codes_lac = ["ARG", "BHS", "BRB", "BLZ", "BOL", "BRA", "CHL", "COL", "CRI", "DOM", "ECU", "SLV", "GTM", "GUY", "HTI", "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER", "SUR", "TTO", "URY", "VEN"]
country_names_lac = ['argentina', 'bahamas', 'barbados', 'belize', 'bolivia', 'brazil', 'chile', 'colombia', 'costa_rica', 'dominican_republic', 'ecuador', 'el_salvador', 'guatemala', 'guyana', 'haiti', 'honduras', 'jamaica', 'mexico', 'nicaragua', 'panama', 'paraguay', 'peru', 'suriname', 'trinidad_and_tobago', 'uruguay', 'venezuela']

correspondece_iso_names = {x:y for x,y in zip(iso3_codes_lac, country_names_lac)}

# Load input data sisepuede_aggregate_calibration_db_20220303.csv
df_input_all_countries = pd.read_csv( os.path.join(data_path, "sisepuede_aggregate_calibration_db_20220303.csv"))

# Define target country
#target_country = sys.argv[1]
target_country = "BRA"

#### Load calib targets by model to run
## Calib bounds
df_calib_targets =  pd.read_csv( os.path.join(data_path, "calib_bounds_sector.csv") )

### Load calib targets
calib_targets_all_sectors =  pd.read_csv( os.path.join(data_path, "calib_targets_all_sectors.csv") )

# Load observed CO2 data
df_co2_observed_data = {"AFOLU" : pd.read_csv(os.path.join(data_path, "emissions_targets_promedios_iso_code3.csv")),
                        "CircularEconomy": pd.read_csv(os.path.join(data_path, "ghg_LAC_circular_ce_iso_code3.csv")),
                        "IPPU" : pd.read_csv(os.path.join(data_path, "ghg_LAC_circular_ippu_iso_code3.csv")),
                        "AllEnergy" : pd.read_csv(os.path.join(data_path, "ghg_LAC_energy_iso_code3.csv"))}


# Define lower and upper time bounds
year_init,year_end = 0,5

# Subset input data
df_input_country = df_input_all_countries.query("iso_code3 =='{}' and (time_period>={} and time_period<={})".format(target_country,year_init,year_end)).reset_index().drop(columns=["index"])

df_input_country_all_time_period = df_input_all_countries.query("iso_code3 =='{}'".format(target_country)).reset_index().drop(columns=["index"])

"""

#### RUN AFOLU CALIBRATION

"""

# Set model to run
models_run = "AFOLU"

# AFOLU calibration targets
calib_targets_afolu = calib_targets_all_sectors.query(f"sector =='{models_run}'")["calib_targets"].reset_index(drop = True)

# AFOLU CO2 observed data
df_co2_observed_data_afolu = df_co2_observed_data[models_run].query("model == '{}' and iso_code3=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init+2014,year_end+2014))


# AFOLU FAO crosswalk co2 - SISEPUEDE
import json

AFOLU_fao_correspondence = json.load(open(os.path.join(data_path,"AFOLU_fao_correspondence.json"), "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}

# Instance CalibrationModel object
calibration = CalibrationModel(year_init+2014, year_end +2014, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets_afolu, df_calib_targets, df_calib_targets, df_input_country_all_time_period,
                                df_co2_observed_data_afolu,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4)

# Test function evaluation with a random calibration vector
X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item()))  for i in calibration.calib_targets["AFOLU"]]

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
plt.savefig(os.path.join(save_data_path, f"{models_run}_{target_country}.png"))
plt.close()

"""

#### RUN CircularEconomy CALIBRATION

"""

models_run = "CircularEconomy"


# CircularEconomy calibration targets
calib_targets_circular_economy = calib_targets_all_sectors.query(f"sector =='{models_run}'")["calib_targets"].reset_index(drop = True)

# CircularEconomy CO2 observed data
df_co2_observed_data_circular_economy = df_co2_observed_data[models_run].query("model == '{}' and iso_code3=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init+2014,year_end+2014))

# Instance of CalibrationModel
calibration_circ_ec = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets_circular_economy, df_calib_targets, df_calib_targets, df_input_country_all_time_period,
                                df_co2_observed_data_circular_economy, cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)

# Test function evaluation with a random calibration vector
X = [np.mean((calibration_circ_ec.df_calib_bounds.loc[calibration_circ_ec.df_calib_bounds["variable"] == i, "min_35"].item(),calibration_circ_ec.df_calib_bounds.loc[calibration_circ_ec.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration_circ_ec.calib_targets["CircularEconomy"]]

# Set calibrated vector for each sector estimated
calibration_circ_ec.set_best_vector("AFOLU",calibration_vector_AFOLU)

# Evaluate random calibration vector in the function to minimize
calibration_circ_ec.f(X)


# Setup optimization algorithm parameters and run calibration with PSO  
parameters_algo = {"alpha" : 0.5, "beta" : 0.8}
calibration_circ_ec.run_calibration("pso", population = 10, maxiter = 10, param_algo = parameters_algo)

# Check fitness
#plt.plot(calibration_circ_ec.fitness_values["CircularEconomy"])
#plt.show()


# Check performance
calibration_vector_CircularEconomy = calibration_circ_ec.best_vector["CircularEconomy"]


with open(os.path.join(save_data_path, "calib_vectors", f'calibration_vector_CircularEconomy_{target_country}.pickle'), 'wb') as f:
    pickle.dump(calibration_vector_CircularEconomy, f)


output_data = calibration_circ_ec.get_output_data(calibration_vector_CircularEconomy, print_sector_model = True)

co2_computed = output_data[calibration_circ_ec.var_co2_emissions_by_sector["CircularEconomy"]].sum(axis=1)
plt.plot(range(year_init,year_end+1),[i/1000 for i in df_co2_observed_data_circular_economy.value],label="historico")
plt.plot(range(year_init,year_end+1),co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
plt.savefig(os.path.join(save_data_path, f"{models_run}_{target_country}.png"))
plt.close()

"""

#### RUN IPPU CALIBRATION

"""

# Set model to run
models_run = "IPPU"

# IPPU calibration targets
calib_targets_ippu = calib_targets_all_sectors.query(f"sector =='{models_run}'")["calib_targets"].reset_index(drop = True)

# IPPU CO2 observed data
df_co2_observed_data_ippu = df_co2_observed_data[models_run].query("model == '{}' and iso_code3=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init+2014,year_end+2014))

# Instance of CalibrationModel
calibration_ippu = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets_ippu, df_calib_targets, df_calib_targets, df_input_country_all_time_period,
                                df_co2_observed_data_ippu, cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)

# Test function evaluation with a random calibration vector
X = [np.mean((calibration_ippu.df_calib_bounds.loc[calibration_ippu.df_calib_bounds["variable"] == i, "min_35"].item(),calibration_ippu.df_calib_bounds.loc[calibration_ippu.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration_ippu.calib_targets["IPPU"]]

# Set calibrated vector for each sector estimated
calibration_ippu.set_best_vector("AFOLU",calibration_vector_AFOLU)
calibration_ippu.set_best_vector("CircularEconomy",calibration_vector_CircularEconomy)
calibration_ippu.set_calib_targets("CircularEconomy", calib_targets_circular_economy)

# Evaluate random calibration vector in the function to minimize
calibration_ippu.f(X)

# Setup optimization algorithm parameters and run calibration with PSO  
parameters_algo = {"alpha" : 0.5, "beta" : 0.8}
calibration_ippu.run_calibration("pso", population = 10, maxiter = 10, param_algo = parameters_algo)

# Check fitness
#plt.plot(calibration_ippu.fitness_values["IPPU"])
#plt.show()

calibration_vector_IPPU = calibration_ippu.best_vector["IPPU"]

with open(os.path.join(save_data_path, "calib_vectors", f'calibration_vector_IPPU_{target_country}.pickle'), 'wb') as f:
    pickle.dump(calibration_vector_IPPU, f)

output_data = calibration_ippu.get_output_data(calibration_vector_IPPU, print_sector_model = True)

co2_computed = output_data[calibration_ippu.var_co2_emissions_by_sector["IPPU"]].sum(axis=1)
plt.plot([i/1000 for i in df_co2_observed_data_ippu.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(f"País : {target_country}. Sector : {models_run}. Calibración integrada")
plt.legend()
plt.savefig(os.path.join(save_data_path, f"{models_run}_{target_country}.png"))
plt.close()

#output_data_ippu = calibration_ippu.get_output_data(calibration_vector_IPPU, print_sector_model = True)

"""

#### RUN AllEnergy CALIBRATION

"""

"""
## Load calibrated vectors

with open(f'calibration_vector_AFOLU_{target_country}.pickle', 'rb') as f:
    calibration_vector_AFOLU = pickle.load(f)

with open(f'calibration_vector_CircularEconomy_{target_country}.pickle', 'rb') as f:
    calibration_vector_CircularEconomy = pickle.load(f)

with open(f'calibration_vector_IPPU_{target_country}.pickle', 'rb') as f:
    calibration_vector_IPPU = pickle.load(f)
"""

# Set model to run
models_run = "AllEnergy"

## Load Energy crosswalk
import json
energy_correspondence = json.load(open(os.path.join(data_path, "energy_subsector_items.json") , "r"))

# AllEnergy calibration targets
calib_targets_energy = calib_targets_all_sectors.query(f"sector =='{models_run}'")["calib_targets"].reset_index(drop = True)


# AllEnergy CO2 observed data
energy_observado = df_co2_observed_data[models_run].query(f"iso_code3=='{target_country}' and (variable >= 2014 and variable <=2019)").reset_index(drop = True)


# Instance of CalibrationModel
calibration_Allenergy = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets_energy, df_calib_targets, df_calib_targets, df_input_country_all_time_period,
                                energy_observado, energy_correspondence, cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)

# Test function evaluation with a random calibration vector
# Set calibrated vector for each sector estimated
calibration_Allenergy.set_best_vector("AFOLU",calibration_vector_AFOLU)

calibration_Allenergy.set_best_vector("CircularEconomy",calibration_vector_CircularEconomy)
calibration_Allenergy.set_calib_targets("CircularEconomy", calib_targets_circular_economy)

calibration_Allenergy.set_best_vector("IPPU",calibration_vector_IPPU)
calibration_Allenergy.set_calib_targets("IPPU", calib_targets_ippu)

output_data_AllEnergy = calibration_Allenergy.get_output_data([1]*len(calib_targets_energy), print_sector_model = True)
#calibrated_data_AllEnergy = calibration_Allenergy.get_calibrated_data([1]*len(calib_targets_energy), print_sector_model = True)

# Test function evaluation
#X = [np.mean((calibration_Allenergy.df_calib_bounds.loc[calibration_Allenergy.df_calib_bounds["variable"] == i, "min_35"].item(),calibration_Allenergy.df_calib_bounds.loc[calibration_Allenergy.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration_Allenergy.calib_targets["AllEnergy"]]
#calibration_Allenergy.f(X)


### ENERGY CALIBRATION
print("INICIA LA CALIBRACION DE ENERGIA")
from optimization_algorithms import PSO

# Tamaño de la población
n = 5
# Número de variables
n_var = len(calib_targets_energy)
l_bounds = np.array(calib_bounds["min_35"])
u_bounds = np.array(calib_bounds["max_35"])
maxiter =  1
# Social scaling parameter
α = 0.8
# Cognitive scaling parameter
β = 0.8
# velocity inertia
w = 0.5

fitness_pso, x_best_pso = PSO(calibration_Allenergy.f, n, maxiter, n_var, l_bounds, u_bounds, α, β, w)

output_data_AllEnergy = calibration_Allenergy.get_output_data(x_best_pso, print_sector_model = True)
calibrated_data_AllEnergy = calibration_Allenergy.get_calibrated_data(x_best_pso, print_sector_model = True)
calibrated_data_AllEnergy["iso_code3"] = target_country
calibrated_data_AllEnergy = calibrated_data_AllEnergy[["time_period","iso_code3"]+[i for i in calibrated_data_AllEnergy.columns if i not in ["time_period","iso_code3"]]]


###############################
###############################
import math

energy_observado = pd.read_csv(os.path.join(data_path, "ghg_LAC_energy_iso_code3.csv"))
energy_observado = energy_observado.query(f"iso_code3=='{target_country}' and (variable >= 2014 and variable <=2019)").reset_index(drop = True)

energy_crosswalk_estimado = {}
energy_crosswalk_observado = {}
energy_crosswalk_error = {}

for subsector, sisepuede_vars in calibration_Allenergy.var_co2_emissions_by_sector["AllEnergy"].items():
    energy_crosswalk_estimado[subsector] = output_data_AllEnergy[sisepuede_vars].sum(1).reset_index(drop = True) 
    energy_crosswalk_observado[subsector] = energy_observado.query(f"subsector_sisepuede == '{subsector}'")[["value"]].sum(1).reset_index(drop = True)
    energy_crosswalk_error[subsector] = (energy_crosswalk_estimado[subsector] - energy_crosswalk_observado[subsector])**2



for k,v in energy_crosswalk_estimado.items():
    plt.plot(energy_crosswalk_estimado[k], label = "Estimado")
    plt.plot(energy_crosswalk_observado[k], label = "Histórico")
    plt.title(f"País : {target_country}. Sector : {models_run}. Subsector : {k}. Calibración integrada")
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(save_data_path, f"{models_run}_{k}_{target_country}.png"))
    plt.close()

#### Save calibrated data
calibrated_data_AllEnergy.to_csv(os.path.join(save_data_path, f"calibrated_data_{target_country}.csv"), index = False)

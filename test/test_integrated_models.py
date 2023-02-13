import os

os.environ['LAC_PATH'] = '/home/milo/Documents/julia/lac_decarbonization'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sisepuede_calibration.calibration_lac import CalibrationModel

data_path = "/home/milo/Documents/egap/SISEPUEDE/packaging_projects/github_projects/sisepuede_calibration_dev/unit_testing/data_test"


iso3_codes_lac = ["ARG", "BHS", "BRB", "BLZ", "BOL", "BRA", "CHL", "COL", "CRI", "DOM", "ECU", "SLV", "GTM", "GUY", "HTI", "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER", "SUR", "TTO", "URY", "VEN"]
country_names_lac = ['argentina', 'bahamas', 'barbados', 'belize', 'bolivia', 'brazil', 'chile', 'colombia', 'costa_rica', 'dominican_republic', 'ecuador', 'el_salvador', 'guatemala', 'guyana', 'haiti', 'honduras', 'jamaica', 'mexico', 'nicaragua', 'panama', 'paraguay', 'peru', 'suriname', 'trinidad_and_tobago', 'uruguay', 'venezuela']

correspondece_iso_names = {x:y for x,y in zip(iso3_codes_lac, country_names_lac)}

#df_input_all_countries = pd.read_csv( os.path.join(data_path, "real_data_2023_01_11.csv"))
df_input_all_countries = pd.read_csv( os.path.join(data_path, "sisepuede_aggregate_calibration_db_20220207.csv"))

# Define target country
target_country = "BRA"

"""

#### RUN AFOLU MODEL

"""

# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
#df_co2_observed_data = pd.read_csv( os.path.join(data_path, "emissions_targets.csv") )
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "emissions_targets_promedios_iso_code3.csv") )
df_co2_observed_data.Nation = df_co2_observed_data.Nation.str.lower()

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
year_init,year_end = 0,5

df_input_country = df_input_all_countries.query("iso_code3 =='{}' and (time_period>={} and time_period<={})".format(target_country,year_init,year_end)).reset_index().drop(columns=["index"])
df_input_country["time_period"] = list(range(1+(year_end-year_init)))


t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and iso_code3=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init+2014,year_end+2014))

df_input_country_all_time_period = df_input_all_countries.query("iso_code3 =='{}'".format(target_country)).reset_index().drop(columns=["index"])

# AFOLU FAO co2
import json
AFOLU_fao_correspondence = json.load(open("/home/milo/Documents/egap/SISEPUEDE/packaging_projects/minimal/AFOLU_fao_correspondence.json", "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}

calibration = CalibrationModel(year_init+2014, year_end +2014, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets, df_calib_targets,df_input_country_all_time_period,
                                df_co2_observed_data,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4)

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item()))  for i in calibration.calib_targets["AFOLU"]]

calibration.f(X)

param_algo = {"alpha" : 0.5, "beta" : 0.8}

calibration.run_calibration("pso", population = 20, maxiter = 10, param_algo = param_algo)

plt.plot(calibration.fitness_values["AFOLU"])
plt.show()

calibration_vector_AFOLU = calibration.best_vector["AFOLU"]

with open('calibration_vector_AFOLU.pickle', 'wb') as f:
    pickle.dump(calibration_vector_AFOLU, f)

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
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "ghg_LAC_circular_ippu_iso_code3.csv") )


calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets_circular_economy = calib_bounds['variable']

df_co2_observed_data = df_co2_observed_data.query("model == '{}' and iso_code3=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init+2014,year_end+2014))

df_co2_observed_data.rename(columns = {"Value" : "value"}, inplace = True)


# Instance of CalibrationModel
calibration_circ_ec = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
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


with open('calibration_vector_CircularEconomy.pickle', 'wb') as f:
    pickle.dump(calibration_vector_CircularEconomy, f)


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
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "ghg_LAC_circular_ippu_iso_code3.csv") )


calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets_ippu = calib_bounds['variable']

remueve_ippu = ['demscalar_ippu_recycled_textiles', 'demscalar_ippu_recycled_glass', 'demscalar_ippu_recycled_plastic', 'demscalar_ippu_recycled_metals', 'demscalar_ippu_recycled_paper', 'demscalar_ippu_recycled_rubber_and_leather', 'demscalar_ippu_recycled_wood']
calib_targets_ippu = calib_targets_ippu[~calib_targets_ippu.isin(remueve_ippu)]

df_co2_observed_data = df_co2_observed_data.query("model == '{}' and iso_code3=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init+2014,year_end+2014))

df_co2_observed_data.rename(columns = {"Value" : "value"}, inplace = True)


# Instance of CalibrationModel
calibration_ippu = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
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

with open('calibration_vector_IPPU.pickle', 'wb') as f:
    pickle.dump(calibration_vector_IPPU, f)

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
calibration_NoEenergy = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
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
calibration_Eenergy = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets_ippu, calib_bounds,df_input_country_all_time_period,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)

calibration_Eenergy.set_best_vector("AFOLU",calibration_vector_AFOLU)

calibration_Eenergy.set_best_vector("CircularEconomy",calibration_vector_CircularEconomy)
calibration_Eenergy.set_calib_targets("CircularEconomy", calib_targets_circular_economy)

calibration_Eenergy.set_best_vector("IPPU",calibration_vector_IPPU)
calibration_Eenergy.set_calib_targets("IPPU", calib_targets_ippu)

output_data_ElectricEnergy = calibration_Eenergy.get_output_data([1], print_sector_model = True)

"""

#### RUN AllEnergy MODEL

"""

## Load calibrated vectors

with open('calibration_vector_AFOLU.pickle', 'rb') as f:
    calibration_vector_AFOLU = pickle.load(f)

with open('calibration_vector_CircularEconomy.pickle', 'rb') as f:
    calibration_vector_CircularEconomy = pickle.load(f)

with open('calibration_vector_IPPU.pickle', 'rb') as f:
    calibration_vector_IPPU = pickle.load(f)

## Load crosswalk
import json
energy_correspondence = json.load(open(os.path.join(data_path, "energy_subsector_items.json") , "r"))

## Load CO2 observation
energy_observado = pd.read_csv(os.path.join(data_path, "ghg_LAC_energy_iso_code3.csv"))
energy_observado = energy_observado.query(f"iso_code3=='{target_country}' and (variable >= 2014 and variable <=2019)").reset_index(drop = True)

# Set model to run
models_run = "AllEnergy"

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_bounds = calib_bounds.query("not(min_35==1 and max_35==1)").reset_index(drop = True)
calib_targets_energy = calib_bounds['variable']



# Instance of CalibrationModel
calibration_Allenergy = CalibrationModel(year_init, year_end, df_input_country, correspondece_iso_names[target_country], models_run,
                                calib_targets_energy, calib_bounds,df_input_country_all_time_period,
                                energy_observado, energy_correspondence, cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4, run_integrated_q = True)

calibration_Allenergy.set_best_vector("AFOLU",calibration_vector_AFOLU)

calibration_Allenergy.set_best_vector("CircularEconomy",calibration_vector_CircularEconomy)
calibration_Allenergy.set_calib_targets("CircularEconomy", calib_targets_circular_economy)

calibration_Allenergy.set_best_vector("IPPU",calibration_vector_IPPU)
calibration_Allenergy.set_calib_targets("IPPU", calib_targets_ippu)

output_data_AllEnergy = calibration_Allenergy.get_output_data([1]*177, print_sector_model = True)
calibrated_data_AllEnergy = calibration_Allenergy.get_calibrated_data([1]*177, print_sector_model = True)

# Test function evaluation
X = [np.mean((calibration_Allenergy.df_calib_bounds.loc[calibration_Allenergy.df_calib_bounds["variable"] == i, "min_35"].item(),calibration_Allenergy.df_calib_bounds.loc[calibration_Allenergy.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration_Allenergy.calib_targets["AllEnergy"]]
calibration_Allenergy.f(X)

########################
#### PRUEBA PSO

# Para hacer el muestreo por Latin Hypecube
from scipy.stats.qmc import LatinHypercube,scale
import math


# Definimos la clase Particle
class Particle:
    def __init__(self,x,v):
        self.x = x
        self.v = v
        self.x_best = x
        
def PSO(f, pop_size, maxiter, n_var, lb, ub, α, β, w):
    '''
    ------------------------------------------
                        PSO
    Particle Swarm Optimization
    -------------------------------------------
    ## Implemented as a minimization algorithm
    # Inputs:
        * f             - function to be minimized
        * pop_size      - number of individuals in the population
        * max_iter     - maximum number of optimization iterations
        * n_var
        * lb
        * ub
        * α             - Social scaling parameter
        * β             - Cognitive scaling parameter
        * w             - velocity inertia
        
    # Output
        * x_best        - best solution found
        * fitness_values - history of best score
    '''   
    # LatinHypercube sampling
    engine = LatinHypercube(d=n_var)
    sample = engine.random(n=pop_size)

    l_bounds = np.array(lb)
    u_bounds = np.array(ub)

    sample_scaled = scale(sample,l_bounds, u_bounds)
    sample_scaled = scale(sample,l_bounds, u_bounds)

    # define particle population
    pob = [Particle(x,np.array([0]*n_var)) for x in sample_scaled]


    
    x_best = pob[0].x_best
    y_best = f(x_best)

    
    # minimum value for the velocity inertia
    w_min = 0.4
    # maximum value for the velocity inertia
    w_max = 0.9

    # Velocidad máxima
    vMax = np.multiply(u_bounds-l_bounds,0.2)
    # Velocidad mínima
    vMin = -vMax

    
    for P in pob:
        y = f(P.x)
        if y < y_best:
            x_best = P.x_best
            y_best = y

    fitness_values = []

    print("*********************")
    print(f"Mejor valor {y_best}")
    print("*********************")
    for k in range(maxiter):
        
        print("-----------------------------")
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("        Iteración {}".format(k))
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("-----------------------------")
        
        for P in pob:
            # Actualiza velocidad de la partícula
            ϵ1,ϵ2 = np.random.uniform(), np.random.uniform()
            P.v = w*P.v + α*ϵ1*(P.x_best - P.x) + β*ϵ2*(x_best - P.x)

            # Ajusta velocidad de la partícula
            index_vMax = np.where(P.v > vMax)
            index_vMin = np.where(P.v < vMin)

            if np.array(index_vMax).size > 0:
                P.v[index_vMax] = vMax[index_vMax]
            if np.array(index_vMin).size > 0:
                P.v[index_vMin] = vMin[index_vMin]

            # Actualiza posición de la partícula
            P.x += P.v

            # Ajusta posición de la particula
            index_pMax = np.where(P.x > u_bounds)
            index_pMin = np.where(P.x < l_bounds)

            if np.array(index_pMax).size > 0:
                P.x[index_pMax] = u_bounds[index_pMax]
            if np.array(index_pMin).size > 0:
                P.x[index_pMin] = l_bounds[index_pMin]

            # Evaluamos la función
            y = f(P.x)

            if y < y_best:
                x_best = np.copy(P.x_best)
                y_best = y
            if y < f(P.x_best):
                P.x_best = np.copy(P.x)
            

            # Actualizamos w

            w = w_max - k * ((w_max-w_min)/maxiter)

        print(y_best)
        fitness_values.append(y_best)

    return fitness_values ,x_best


# Tamaño de la población
n = 5
# Número de variables
n_var = len(calib_targets_energy)
l_bounds = np.array(calib_bounds["min_35"])
u_bounds = np.array(calib_bounds["max_35"])
maxiter =  5
# Social scaling parameter
α = 0.8
# Cognitive scaling parameter
β = 0.8
# velocity inertia
w = 0.5

fitness_pso, x_best_pso = PSO(calibration_Allenergy.f, n, maxiter, n_var, l_bounds, u_bounds, α, β, w)
x_best_pso


"""
##################################

param_algo = {"precision" : 6, "pc" : 0.9}

calibration_Allenergy.run_calibration("genetic_binary", population = 20, maxiter = 10, param_algo = param_algo)

plt.plot(calibration_Allenergy.fitness_values["AllEnergy"])
plt.show()

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

co2_df = pd.DataFrame(energy_crosswalk_error)
calibration.error_by_item = co2_df
calibration.error_by_sector_energy = energy_crosswalk_error
co2_df_total = co2_df.sum(1)

co2_df_observado = pd.DataFrame(energy_crosswalk_observado)

ponderadores = (co2_df_observado.mean().abs()/co2_df_observado.mean().abs().sum()).apply(math.exp)   
co2_df_total = (ponderadores*co2_df).sum(1)

if calibration_Allenergy.cv_calibration:

    co2_df_total = [co2_df_total[i] for i in calibration_Allenergy.cv_training]
    output = np.sum(co2_df_total)

else:
    output = np.sum(co2_df_total)

"""
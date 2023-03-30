import os

os.environ['LAC_PATH'] = '/home/milo/Documents/egap/calibration/lac_decarbonization'

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sisepuede_calibration.calibration_lac import CalibrationModel

# Para hacer el muestreo por Latin Hypecube
from scipy.stats.qmc import LatinHypercube,scale


from request_server.request_server import send_request_py


# Set directories
#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.getcwd()
data_path = os.path.abspath(os.path.join(dir_path,"..","data","inputs_sisepuede" ))
pickle_path = os.path.abspath(os.path.join(dir_path,"..","output","calib_vectors" ))
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

#### RUN AllEnergy CALIBRATION

"""

## Load calibrated vectors

with open(os.path.join (pickle_path, f'calibration_vector_AFOLU_{target_country}.pickle'), 'rb') as f:
    calibration_vector_AFOLU = pickle.load(f)

with open(os.path.join (pickle_path, f'calibration_vector_CircularEconomy_{target_country}.pickle'), 'rb') as f:
    calibration_vector_CircularEconomy = pickle.load(f)

with open(os.path.join (pickle_path, f'calibration_vector_IPPU_{target_country}.pickle'), 'rb') as f:
    calibration_vector_IPPU = pickle.load(f)

## Load calib targets
calib_targets_circular_economy = calib_targets_all_sectors.query(f"sector =='CircularEconomy'")["calib_targets"].reset_index(drop = True)
calib_targets_ippu = calib_targets_all_sectors.query(f"sector =='IPPU'")["calib_targets"].reset_index(drop = True)

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

## output_data_AllEnergy = calibration_Allenergy.get_output_data([1]*len(calib_targets_energy), print_sector_model = True)

### ENERGY CALIBRATION
print("INICIA LA CALIBRACION DE ENERGIA")

# Tamaño de la población
n = 1

# Número de variables
n_var = len(calib_targets_energy)

# Muestreamos por LHC
engine = LatinHypercube(d=n_var)
sample = engine.random(n=n)

# Definimos límites superiores e inferiores
l_bounds = calibration_Allenergy.df_calib_bounds[calibration_Allenergy.df_calib_bounds.variable.isin(calib_targets_energy)]["min_35"].to_numpy()
u_bounds = calibration_Allenergy.df_calib_bounds[calibration_Allenergy.df_calib_bounds.variable.isin(calib_targets_energy)]["max_35"].to_numpy()

sample_scaled = scale(sample,l_bounds, u_bounds)


# Máxima cantidad de iteraciones por individuo
maxiter =  5

# Social scaling parameter
α = 0.8
# Cognitive scaling parameter
β = 0.8
# velocity inertia
w = 0.5

# velocity inertia
w = 0.5
# minimum value for the velocity inertia
w_min = 0.4
# maximum value for the velocity inertia
w_max = 0.9


# Velocidad máxima
vMax = np.multiply(u_bounds-l_bounds,0.2)
# Velocidad mínima
vMin = -vMax

# Definimos la clase Particle
class Particle:
    def __init__(self,x,v):
        self.x = x
        self.v = v
        self.x_best = x


IP_SERVER_ADD = sys.argv[1]
OPTIMIZATION_STAGE = sys.argv[2]
ID_IND = sys.argv[3]


if OPTIMIZATION_STAGE == 'INIT_POP':

    print(OPTIMIZATION_STAGE)
    particula = Particle(sample_scaled[0], np.array([0]*n_var))

    value_f = calibration_Allenergy.f(particula.x)

    best_global_vector = send_request_py(IP_SERVER_ADD, value_f, particula.x)
    with open(f'PSO_x_best_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(particula.x_best, f)

    with open(f'PSO_x_pos_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(particula.x, f)

    with open(f'PSO_velocity_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(particula.v, f)

    with open(f'PSO_value_x_best_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(value_f, f)

    with open(f'PSO_k_it_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(0, f)

elif OPTIMIZATION_STAGE == 'EVOLUTIVE_CYCLE':
    with open(f'PSO_x_best_{ID_IND}.pickle', 'rb') as f:
        x_best = pickle.load(f)

    with open(f'PSO_x_pos_{ID_IND}.pickle', 'rb') as f:
        x_pos = pickle.load(f)

    with open(f'PSO_velocity_{ID_IND}.pickle', 'rb') as f:
        velocity = pickle.load(f)

    with open(f'PSO_value_x_best_{ID_IND}.pickle', 'rb') as f:
        value_x_best = pickle.load(f)

    with open(f'PSO_k_it_{ID_IND}.pickle', 'rb') as f:
        k_it = pickle.load(f)

    print(f"{OPTIMIZATION_STAGE}\nIteration {k_it}")

    particula = Particle(x_pos, velocity)

    particula.x_best = np.copy(x_best)

    best_global_vector = send_request_py(IP_SERVER_ADD, 10000000, []) 

    # Actualiza velocidad de la partícula
    ϵ1,ϵ2 = np.random.uniform(), np.random.uniform()
    particula.v = w*particula.v + α*ϵ1*(particula.x_best - particula.x) + β*ϵ2*(best_global_vector - particula.x)

    # Ajusta velocidad de la partícula
    index_vMax = np.where(particula.v > vMax)
    index_vMin = np.where(particula.v < vMin)

    if np.array(index_vMax).size > 0:
        particula.v[index_vMax] = vMax[index_vMax]
    if np.array(index_vMin).size > 0:
        particula.v[index_vMin] = vMin[index_vMin]

    # Actualiza posición de la partícula
    particula.x += particula.v

    # Ajusta posición de la particula
    index_pMax = np.where(particula.x > u_bounds)
    index_pMin = np.where(particula.x < l_bounds)

    if np.array(index_pMax).size > 0:
        particula.x[index_pMax] = u_bounds[index_pMax]
    if np.array(index_pMin).size > 0:
        particula.x[index_pMin] = l_bounds[index_pMin]

    # Evaluamos la función
    y = calibration_Allenergy.f(particula.x)

    best_global_vector = send_request_py(IP_SERVER_ADD, y, particula.x)


    if not all(np.array(best_global_vector)==particula.x):
        if y < value_x_best:
            particula.x_best = np.copy(particula.x)
            value_x_best = y

    # Actualizamos w

    w = w_max - k_it * ((w_max-w_min)/maxiter)

    k_it += k_it
     
    with open(f'PSO_x_best_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(particula.x_best, f)

    with open(f'PSO_x_pos_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(particula.x, f)

    with open(f'PSO_velocity_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(particula.v, f)

    with open(f'PSO_value_x_best_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(value_x_best, f)
    
    with open(f'PSO_k_it_{ID_IND}.pickle', 'wb') as f:
        pickle.dump(k_it, f)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel,Socioeconomic,sa

import warnings

warnings.filterwarnings("ignore")

df_input_all_countries = pd.read_csv("real_data_2022_10_04.csv")

# Define target country
import sys 
#target_country = "brazil"
target_country = sys.argv[1]
# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/emissions_targets.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_bounds/output/calib_bounds_sector.csv")

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

calib_bounds_groups = calib_bounds.groupby("group")
indices_params = list(calib_bounds_groups.groups[0])

for i,j in calib_bounds_groups.groups.items():
    if i!=0:
        indices_params.append(j[0])

calib_targets = calib_bounds['variable'].iloc[indices_params].reset_index(drop=True)

# Define lower and upper time bounds
year_init,year_end = 2014,2019

df_input_country = df_input_all_countries.query("Nation =='{}' and (Year>={} and Year<={})".format(target_country,year_init,year_end)).reset_index().drop(columns=["index"])
df_input_country["time_period"] = list(range(1+(year_end-year_init)))

df_input_country_all_time_period = df_input_all_countries.query("Nation =='{}'".format(target_country)).reset_index().drop(columns=["index"])

t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and Nation=='{}' and (Year >= {} and Year <= {} )".format(models_run,target_country,year_init,year_end))
df_co2_observed_data =  df_co2_observed_data

# AFOLU FAO co2
import json
AFOLU_fao_correspondence = json.load(open("build_CO2_data_models/FAO_correspondence/AFOLU_fao_correspondence.json", "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}


acumula_co2_it = {i:{} for i in range(2,4)}
N = 5

for prec in range(2,4):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"             {prec}              ")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    for i in range(N):
        print("-------------------------------------")
        print(f"             {i}              ")
        print("-------------------------------------")

        calibration = CalibrationModel(df_input_country, target_country, models_run,
                                        calib_targets, calib_bounds,df_input_country_all_time_period,
                                        df_co2_observed_data,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=prec)

        calibration.run_calibration("pso", population = 100, maxiter = 10)


        output_data = calibration.get_output_data(calibration.best_vector["AFOLU"])

        item_val_afolu = {}
        observed_val_afolu = {}
        for item, vars in AFOLU_fao_correspondence.items():
            if vars:
                item_val_afolu[item] = output_data[vars].sum(1).to_list()
                observed_val_afolu[item] = (df_co2_observed_data.query("Item_Code=={}".format(item)).Value/1000).to_list()

        observed_val_afolu = {k:v for k,v in observed_val_afolu.items() if len(v) > 0}

        acumula_co2_it[prec][i] = pd.DataFrame(item_val_afolu).sum(axis=1)
        co2_historical = pd.DataFrame(observed_val_afolu).sum(axis=1)

acumula_ejecuciones = {}
for i in range(2,4):
    acumula = []
    for k,v in acumula_co2_it[i].items():
        acumula.append(v)
    acumula_ejecuciones[f"mean_{i}"] = np.array(acumula).mean(0)
    acumula_ejecuciones[f"std_{i}"] = np.array(acumula).std(0)

df = pd.DataFrame(acumula_ejecuciones)

for i in range(2,4):
    plt.errorbar(range(2014,2020),df[f"mean_{i}"], df[f"std_{i}"], label = i)
plt.plot(range(2014,2020),co2_historical,label = "historico")
plt.legend()
plt.savefig(f"output_calib/{target_country}.png")


df_prec_acumula = []
for k,v in acumula_co2_it.items():
    print(f"----------{k}-------")
    df_prec = pd.DataFrame(v)
    df_prec["historico"] = co2_historical
    df_prec["Nation"] = target_country
    df_prec["precision"]= k
    df_prec_acumula.append(df_prec)
df_prec_acumula = pd.concat(df_prec_acumula, ignore_index = True)

df_prec_acumula.to_csv(f"output_calib/co2_raw_{target_country}.csv", index = False)

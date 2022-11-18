import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel,Socioeconomic,sa

import warnings

warnings.filterwarnings("ignore")

df_input_all_countries = pd.read_csv("real_data_2022_10_04.csv")

# Define target country
target_country = "brazil"

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
#calib_bounds = calib_bounds.query("variable !='pij_calib'")

calib_bounds_groups = calib_bounds.groupby("group")
indices_params = list(calib_bounds_groups.groups[0])

for i,j in calib_bounds_groups.groups.items():
    if i!=0:
        indices_params.append(j[0])

calib_targets = calib_bounds['variable'].iloc[indices_params].reset_index(drop=True)
calib_targets = calib_targets.append(pd.Series("pij_calib"),ignore_index=True)


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
AFOLU_fao_correspondence = json.load(open("build_CO2_data_models/FAO_correspondence/AFOLU_fao_correspondence.json", "r"))
AFOLU_fao_correspondence = {k:v for k,v in AFOLU_fao_correspondence.items() if v}

calibration = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets, calib_bounds,df_input_country_all_time_period, 
                                df_co2_observed_data,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4)

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item()))  for i in calibration.calib_targets["AFOLU"]]

calibration.f(X)

param_algo = {"alpha" : 0.5, "beta" : 0.8}

calibration.run_calibration("pso", population = 10, maxiter = 10, param_algo = param_algo)

param_algo = {"precision" : 6, "pc" : 0.8}
calibration.run_calibration("genetic_binary", population = 10, maxiter = 10, param_algo = param_algo)

plt.plot(calibration.fitness_values["AFOLU"])
plt.show()


output_data = calibration.get_output_data(calibration.best_vector["AFOLU"])
calibration.build_bar_plot_afolu(calibration.best_vector["AFOLU"], show = True)

AFOLU_fao_correspondence = json.load(open("build_CO2_data_models/FAO_correspondence/AFOLU_fao_correspondence.json", "r"))
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
plt.title(target_country)
plt.legend()
plt.show()


import json


AFOLU_fao_correspondence = json.load(open("build_CO2_data_models/FAO_correspondence/AFOLU_fao_correspondence.json", "r"))
item_val_afolu = {}

AFOLU_fao_correspondence_min = {k:v for k,v in AFOLU_fao_correspondence.items() if k in ["5058","6750"]}

for item, vars in AFOLU_fao_correspondence_min.items():
    if vars:
        item_val_afolu[item] = output_data[vars].sum(1).to_list()

for item,values in item_val_afolu.items():
    plt.plot(values,label = item)
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



calib_bounds.loc[calib_bounds["variable"].isin([i for i in calib_bounds["variable"] if "pij" in i]),"max_35"] = 1.5
calib_bounds.loc[calib_bounds["variable"].isin([i for i in calib_bounds["variable"] if "pij" in i]),"min_35"] = 0.5

acumula_co2_it = {}

for i in range(2,7):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"             {i}              ")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    calibration = CalibrationModel(df_input_country, target_country, models_run,
                                    calib_targets, calib_bounds,df_input_country_all_time_period,
                                    df_co2_observed_data,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=i)

    calibration.run_calibration("pso", population = 10, maxiter = 10)


    output_data = calibration.get_output_data(calibration.best_vector["AFOLU"])

    item_val_afolu = {}
    observed_val_afolu = {}
    for item, vars in AFOLU_fao_correspondence.items():
        if vars:
            item_val_afolu[item] = output_data[vars].sum(1).to_list()
            observed_val_afolu[item] = (df_co2_observed_data.query("Item_Code=={}".format(item)).Value/1000).to_list()

    observed_val_afolu = {k:v for k,v in observed_val_afolu.items() if len(v) > 0}

    acumula_co2_it[i] = pd.DataFrame(item_val_afolu).sum(axis=1)
    co2_historical = pd.DataFrame(observed_val_afolu).sum(axis=1)

for k,v in acumula_co2_it.items():
    plt.plot(v, label = f"Estimado . Prec {k}")   
plt.plot(co2_historical,label="estimado")
plt.title(target_country)
plt.legend()
plt.show()
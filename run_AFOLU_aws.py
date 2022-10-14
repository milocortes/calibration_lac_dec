import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel,Socioeconomic,sa

import warnings

warnings.filterwarnings("ignore")

df_input_all_countries = pd.read_csv("real_data_2022_10_04.csv")

# Define target country
import sys 
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

"""
Calibración todos los países
"""

acumula_vectores = []
acumula_simulados_observados = []
acumula_percent_diff = []

for i in range(5):
    print(i)
    calibration = CalibrationModel(df_input_country, target_country, models_run,
                                    calib_targets, calib_bounds,df_input_country_all_time_period,
                                    df_co2_observed_data,AFOLU_fao_correspondence,cv_training = [0,1,2,3,4,5] ,cv_calibration = False,precition=4)

    calibration.run_calibration("pso", population = 100, maxiter = 40)

    calib_vec_to_df = pd.DataFrame(np.array(calibration.best_vector["AFOLU"])[np.newaxis,:],columns=list(calib_targets))

    agrupa = calib_bounds.groupby("group")
    group_list = calib_bounds["group"].unique()
    total_groups = len(group_list)

    for group in group_list:
        group = int(group)
        if group == 0:
            index_var_group = calib_bounds["variable"].iloc[agrupa.groups[group]]
            calib_vec_to_df[index_var_group] =  calib_vec_to_df[index_var_group]
        else:
            index_var_group = calib_bounds["variable"].iloc[agrupa.groups[group]]
            for col in index_var_group:
                calib_vec_to_df[col] =  calib_vec_to_df[index_var_group.iloc[0]]

    error = calibration.f(calibration.best_vector["AFOLU"])
    acumula_percent_diff.append(calibration.percent_diff.to_numpy())
    
    df_error = pd.DataFrame({"error":[error]})
    calib_vec_to_df = pd.concat([calib_vec_to_df,df_error],axis = 1)


    acumula_vectores.append(calib_vec_to_df)

    output_data = calibration.get_output_data(calibration.best_vector["AFOLU"])

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
    pd_computed_historical = pd.DataFrame({"country" : [target_country]*6,"iteration":[i]*6,"year":range(2014,2020),"simulado": co2_computed})
    acumula_simulados_observados.append(pd_computed_historical)



all_calib_vec_to_df = pd.concat(acumula_vectores,ignore_index=True)
all_calib_vec_to_df["nation"] = target_country
all_calib_vec_to_df["iteration"] = range(5)
df_acumula_simulados_observados = pd.concat(acumula_simulados_observados,ignore_index=True)
historicos = pd.DataFrame({"country" : [target_country]*6,"iteration":[i]*6,"year":range(2014,2020),"historico": co2_historical})

all_calib_vec_to_df.to_csv("output_calib/AFOLU/calib_vec_AFOLU_{}.csv".format(target_country),index=False)
df_acumula_simulados_observados.to_csv("output_calib/AFOLU/co2_simulados_AFOLU_{}.csv".format(target_country),index=False)
historicos.to_csv("output_calib/AFOLU/co2_historicos_AFOLU_{}.csv".format(target_country),index=False)

acumula_percent_diff_np = np.array(acumula_percent_diff)
acumula_percent_diff_np_mean = np.mean(acumula_percent_diff_np,axis=1)
df_acumula_percent_diff_np = pd.DataFrame(acumula_percent_diff_np_mean,columns = list(calibration.percent_diff.columns))
df_acumula_percent_diff_np.to_csv("output_calib/AFOLU/percent_diff_AFOLU_{}.csv".format(target_country),index=False)

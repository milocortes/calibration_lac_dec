import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel,Socioeconomic,sa

import warnings

warnings.filterwarnings("ignore")

models_run = 'AFOLU'
df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

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


countries_list = ['argentina','bahamas','barbados','belize','brazil','bolivia','chile','colombia','costa_rica','dominican_republic','ecuador','el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']

agrupa = calib_bounds.groupby("group")
group_list = calib_bounds["group"].unique()
total_groups = len(group_list)

new_df_input_all_countries = []

for country in countries_list:
    print(country)
    df_input_data = df_input_all_countries.query(f"country=='{country}'")
    df_calib_country = pd.read_csv(f'output_calib/AFOLU/calib_vec_AFOLU_{country}.csv')
    df_calib_country = df_calib_country.groupby("nation").min()

    for k,v in list(df_calib_country.to_dict().items())[:-2]:
        val = v[country]
        df_input_data[k] = df_input_data[k]*val
    new_df_input_all_countries.append(df_input_data)

new_df_input_all_countries = pd.concat(new_df_input_all_countries)

new_df_input_all_countries.to_csv("calibrated_fake_data_complete.csv",index=False)


### time mpiexec  --oversubscribe -np 3 python calibration_models.py
import os, os.path
import numpy as np
import pandas as pd
from utils import *

os.chdir("lac_decarbonization/python")
import data_structures as ds
import setup_analysis as sa
import support_functions as sf
import sector_models as sm
import argparse

from model_socioeconomic import Socioeconomic
from model_afolu import AFOLU
from model_circular_economy import CircularEconomy
from model_ippu import IPPU

import skopt
from skopt import forest_minimize

import itertools
from mpi4py import MPI

import warnings
import datetime

warnings.filterwarnings("ignore")

import glob


# Load input data: fake_data_complete.csv
df_input_data_path = os.path.join(sa.dir_ref, "fake_data", "fake_data_complete.csv")
df_input_data = pd.read_csv(df_input_data_path)
df_input_data = df_input_data.loc[0:8]
# Set model to run
models_run = "CircularEconomy"

'''
 Which input file we will be using to iterate over the model?
'''
os.chdir("../..")
# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

'''
 Run optimization engine
 which variables we are calibrating?
'''
# Load calib targets by model to run
df_calib_targets =   pd.read_csv("build_CO2_data_models/output/df_calib_targets_models.csv")

# All params
all_params = df_input_data.columns
# Rest of params
calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
rest_parameters = list(set(all_params) -set(calib_targets))
rest_parameters_df = df_input_data[rest_parameters]

'''
Add observed data to df_input_data
'''
# For each country make a copy of df_input_data
all_countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador',
 'el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']

df_input_data_countries = pd.DataFrame()

for country in all_countries_list:
    partial = df_input_data
    partial["country"] = country
    df_input_data_countries = pd.concat([df_input_data_countries,partial])

# Load csv for specific model
#list_csv_obs_data =  [i.split("/")[-1][:-4] for i in glob.glob("observed_data/{}/*.csv".format(models_run))]
list_csv_obs_data =  [i.split("/")[-1][:-4] for i in glob.glob("observed_data/*.csv".format(models_run))]

excluye = ["population_rural","population_urban","va_commercial_mmm_usd","va_industrial_mmm_usd","va_manufacturing_mmm_usd","va_mining_mmm_usd"]
# Recollect observed data
dic_validate_category = {"ww_industrial" : [], "frac_wali_ww_domestic_rural_treatment_path" : [], "frac_wali_ww_domestic_urban_treatment_path" : []}

for i in list(set(list_csv_obs_data) -set(excluye)):
    if i in rest_parameters:
        if "ww_industrial" in i:
            dic_validate_category["ww_industrial"].append(i)
        if "frac_wali_ww_domestic_rural_treatment_path" in i:
            dic_validate_category["frac_wali_ww_domestic_rural_treatment_path"].append(i)
        if "frac_wali_ww_domestic_urban_treatment_path" in i:
            dic_validate_category["frac_wali_ww_domestic_urban_treatment_path"].append(i)

        print("Adding {} data".format(i))
        #df_csv_obs_data = pd.read_csv("observed_data/{}/{}.csv".format(models_run,i))
        df_csv_obs_data = pd.read_csv("observed_data/{}.csv".format(i))
        df_csv_obs_data["Nation"] = df_csv_obs_data["Nation"].apply(lambda x: x.lower().replace(" ","_"))
        country_in_observed_data = list(df_csv_obs_data["Nation"])
        country_without_observed_data = list(set(all_countries_list) - set(country_in_observed_data))

        for country in country_in_observed_data:
            mean_value_obs = np.mean(df_csv_obs_data.query("Nation=='{}'".format(country))[i])
            '''
            if "frac" in i:
                if mean_value_obs > 1:
                    mean_value_obs = mean_value_obs/100
            '''
            df_input_data_countries.loc[df_input_data_countries['country'] == country, i] = mean_value_obs

        if country_without_observed_data:
            mean_value_obs = np.mean(df_csv_obs_data[i])
            for country in country_without_observed_data:
                df_input_data_countries[i][df_input_data_countries['country'] == country] = mean_value_obs

dic_validate_category["frac_wali_ww_domestic_rural_treatment_path"] += ["frac_wali_ww_domestic_rural_treatment_path_aerobic","frac_wali_ww_domestic_rural_treatment_path_anaerobic"]
dic_validate_category["frac_wali_ww_domestic_urban_treatment_path"] += ["frac_wali_ww_domestic_urban_treatment_path_aerobic","frac_wali_ww_domestic_urban_treatment_path_anaerobic"]

# Verify group values restrictions
partial_df_input_data_countries = pd.DataFrame()

for country in country_in_observed_data:
    country_partial_df_input_data_countries = df_input_data_countries.query("country =='{}'".format(country))
    for category,variables in dic_validate_category.items():
        total = sum(country_partial_df_input_data_countries[variables].loc[0])
        for var in variables:
            country_partial_df_input_data_countries[var] = country_partial_df_input_data_countries[var].apply(lambda x: x/total)

    partial_df_input_data_countries = pd.concat([partial_df_input_data_countries,country_partial_df_input_data_countries])

# Add variable that change

countries_list = ['argentina','bahamas','barbados','belize','brazil','chile','colombia',
 'costa_rica','dominican_republic','ecuador','el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama',
 'paraguay','peru','uruguay']

for var in excluye:
    print("Adding {}".format(var))
    df_var_change = pd.read_csv("observed_data/{}.csv".format(var))

    for country in countries_list:
        df_var_change_country = df_var_change.query("Nation == '{}'".format(country))
        partial_df_input_data_countries.loc[partial_df_input_data_countries["country"]==country, var] = df_var_change_country[var].to_numpy()

df_input_data_countries = partial_df_input_data_countries

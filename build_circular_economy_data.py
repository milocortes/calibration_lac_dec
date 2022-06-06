
### time mpiexec  --oversubscribe -np 3 python calibration_models.py
import os, os.path
cwd = os.getcwd()
import sys
sys.path.append(cwd + '/lac_decarbonization/python')
import numpy as np
import pandas as pd
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
import glob

warnings.filterwarnings("ignore")



# Load input data: fake_data_complete.csv
df_input_data_path = os.path.join(sa.dir_ref, "fake_data", "fake_data_complete.csv")
df_input_data = pd.read_csv(df_input_data_path)
df_input_data = df_input_data.loc[0:8]

# Set model to run
models_run = "CircularEconomy"

'''
 Which input file we will be using to iterate over the model?
'''
#os.chdir("../..")
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

# calibration bounds
calib_bounds = pd.read_csv("bounds/model_input_variables_ce_demo.csv")
calib_bounds = calib_bounds[["variable","min_35","max_35"]]

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

#for i in list_csv_obs_data:
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

#countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador',
# 'el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']
countries_list = ['argentina']

for var in excluye:
    print("Adding {}".format(var))
    df_var_change = pd.read_csv("observed_data/{}.csv".format(var))

    for country in countries_list:
        df_var_change_country = df_var_change.query("Nation == '{}'".format(country))
        partial_df_input_data_countries.loc[partial_df_input_data_countries["country"]==country, var] = df_var_change_country[var].to_numpy()


df_input_data_countries = partial_df_input_data_countries

path_csv = "output_calib/{}/".format(models_run)
df_afolu_all_countries = pd.DataFrame()

# Define lower and upper time bounds

year_init,year_end = 2006,2014

df_csv_all_to_save = pd.DataFrame()

for  target_country in countries_list:
    print("Colllect results {}".format(target_country))
    target_country_ref = target_country.lower().replace(" ","_")

    df_csv_all = pd.DataFrame()
    df_csv_all_mean = pd.DataFrame()

    for i in range(2):
        df_csv = pd.read_csv(path_csv+"mpi_output/{}_cv_{}_mpi_{}.csv".format(models_run,target_country_ref,i))
        df_csv_all = pd.concat([df_csv_all,df_csv[calib_targets]])
        df_csv_all_mean = pd.concat([df_csv_all_mean,df_csv])

    df_csv_all.reset_index(drop=True,inplace=True)
    df_csv_all_mean.reset_index(drop=True,inplace=True)
    df_csv_all_mean = pd.DataFrame(df_csv_all_mean.mean().to_dict(), index=[0])
    df_csv_all_mean["country"] = target_country
    df_csv_all_to_save = pd.concat([df_csv_all_to_save,df_csv_all_mean])
    df_csv_all.to_csv(path_csv+"processing_output/cv_{}_calibration.csv".format(target_country_ref),index = False)
    df_afolu_country = pd.DataFrame()

    for i in range(df_csv_all.shape[0]):
        x = df_csv_all.loc[[i]].values[0]
        calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
        x_mean = df_input_data_countries.query("country =='{}'".format(target_country))[calib_targets].mean() * x
        b1 = pd.DataFrame.from_dict({j:[i]*9 for i,j in zip(x_mean,calib_targets)})
        partial_rest_parameters_df = df_input_data_countries.query("country =='{}'".format(target_country))[rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)
        model_circular_economy = CircularEconomy(sa.model_attributes)
        df_model_data_project = model_circular_economy.project(input_pivot)

        out_vars = ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"]
        model_data_co2e = df_model_data_project[out_vars].sum(axis=1)

        df_afolu_partial = pd.DataFrame.from_dict({"time":[t for t in range(9)],"area":[target_country_ref]*9 ,"value": model_data_co2e,"id" : [i]*9,"type" :["simulation"]*9})
        df_afolu_country = pd.concat([df_afolu_country,df_afolu_partial])

    co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))
    trend = co2_observed_data.value
    trend = [i/1000 for i in trend]

    df_afolu_historical = pd.DataFrame.from_dict({"time":[t for t in range(9)], "area":[target_country_ref]*9 ,"value": trend,"id" : [0]*9,"type" :["historical"]*9})
    df_afolu_country = pd.concat([df_afolu_country,df_afolu_historical])

    df_afolu_all_countries = pd.concat([df_afolu_all_countries,df_afolu_country])

df_csv_all_to_save.to_csv(path_csv+"calib_vector_circular_economy_all_countries.csv",index = False)
df_afolu_all_countries.to_csv(path_csv+"cv_results_circular_economy_all_countries.csv",index = False)

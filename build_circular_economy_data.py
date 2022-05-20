
### time mpiexec  --oversubscribe -np 3 python calibration_models.py
import os, os.path
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

warnings.filterwarnings("ignore")


def get_cv_index(k):
    lst_all = list(itertools.product([0, 1], repeat=7))
    lst_sub = []

    for i in lst_all:
        if sum(i)==k:
            lst_sub.append(i)

    training = []
    test = []

    for byte in lst_sub:
        index_list = [0]
        for k,i in enumerate(list(byte)):
            if i ==1:
                index_list.append(k+1)
        index_list.append(8)
        training.append(index_list)
        test.append(list(set([i for i in range(9)]) - set(index_list)))
    return training,test


# Load input data: fake_data_complete.csv
df_input_data_path = os.path.join(sa.dir_ref, "fake_data", "fake_data_complete.csv")
df_input_data = pd.read_csv(df_input_data_path)

# Set model to run
models_run = "CircularEconomy"

'''
 Which input file we will be using to iterate over the model?
'''
# Load observed CO2 data
df_co2_observed_data = pd.read_csv("../build_CO2_data_models/build_CO2_data_models/output/co2_all_models.csv")

'''
 Run optimization engine
 which variables we are calibrating?
'''
# Load calib targets by model to run
df_calib_targets =   pd.read_csv("../build_CO2_data_models/build_CO2_data_models/output/df_calib_targets_models.csv")

# All params
all_params = df_input_data.columns
# Rest of params
calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
rest_parameters = list(set(all_params) -set(calib_targets))
rest_parameters_df = df_input_data[rest_parameters]


countries = ["argentina","bahamas","barbados","belize","brazil","chile","colombia","costa_rica","dominican_republic",
"ecuador","el_salvador","guatemala","guyana","haiti","honduras","jamaica","mexico","nicaragua","panama","paraguay","peru","uruguay"]

path_csv = "../output_calib/"
df_afolu_all_countries = pd.DataFrame()

# Define lower and upper time bounds

year_init,year_end = 2006,2014


for  target_country in countries:
    print("Colllect results {}".format(target_country))
    target_country_ref = target_country.lower().replace(" ","_")

    df_csv_all = pd.DataFrame()

    for i in range(8):
        df_csv = pd.read_csv(path_csv+"{}_cv_{}_mpi_{}.csv".format(models_run,target_country_ref,i))
        df_csv_all = pd.concat([df_csv_all,df_csv[calib_targets]])

    df_csv_all.reset_index(drop=True,inplace=True)
    df_csv_all.to_csv(path_csv+"cv_{}_calibration.csv".format(target_country_ref),index = False)
    df_afolu_country = pd.DataFrame()

    for i in range(df_csv_all.shape[0]):
        x = df_csv_all.loc[[i]].values[0]
        calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
        x_mean = df_input_data[calib_targets].mean() * x
        b1 = pd.DataFrame.from_dict({j:[i]*9 for i,j in zip(x_mean,calib_targets)})
        input_pivot =  pd.concat([b1.reset_index(drop=True), rest_parameters_df], axis=1)
        model_circular_economy = CircularEconomy(sa.model_attributes)
        df_model_data_project = model_circular_economy.project(input_pivot)

        out_vars = df_model_data_project.columns[ ["emission_co2e" in i for i in  df_model_data_project.columns]]
        model_data_co2e = df_model_data_project[out_vars].sum(axis=1)

        df_afolu_partial = pd.DataFrame.from_dict({"time":[t for t in range(9)],"area":[target_country_ref]*9 ,"value": model_data_co2e,"id" : [i]*9,"type" :["simulation"]*9})
        df_afolu_country = pd.concat([df_afolu_country,df_afolu_partial])

    co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))
    trend = co2_observed_data.value
    trend = [i/1000 for i in trend]

    df_afolu_historical = pd.DataFrame.from_dict({"time":[t for t in range(9)], "area":[target_country_ref]*9 ,"value": trend,"id" : [0]*9,"type" :["historical"]*9})
    df_afolu_country = pd.concat([df_afolu_country,df_afolu_historical])

    df_afolu_all_countries = pd.concat([df_afolu_all_countries,df_afolu_country])

df_afolu_all_countries.to_csv(path_csv+"cv_results_circular_economy_all_countries.csv",index = False)

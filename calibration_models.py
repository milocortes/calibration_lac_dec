### time mpiexec  --oversubscribe -np 3 python calibration_models.py
import os, os.path
import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
#os.chdir("lac_decarbonization/python")
cwd = os.getcwd()

sys.path.append(cwd + '/lac_decarbonization/python')

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


# Define target country
target_country = str(sys.argv[1])
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

countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador','el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']
## All countries
# {'argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador','el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela'}


for var in excluye:
    print("Adding {}".format(var))
    df_var_change = pd.read_csv("observed_data/{}.csv".format(var))

    for country in countries_list:
        if country != "venezuela":
            df_var_change_country = df_var_change.query("Nation == '{}'".format(country))
            partial_df_input_data_countries.loc[partial_df_input_data_countries["country"]==country, var] = df_var_change_country[var].to_numpy()


df_input_data_countries = partial_df_input_data_countries

# Define lower and upper time bounds
year_init,year_end = 2006,2014

#######################################
#### Genetic + Decision trees


'''
Define objetive function
'''

def objective_a(params):
    if models_run == "AFOLU":
        calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
        x_mean = df_input_data_countries.query("country =='{}'".format(target_country))[calib_targets].mean() * params
        b1 = pd.DataFrame.from_dict({j:[i]*9 for i,j in zip(x_mean,calib_targets)})
        partial_rest_parameters_df = df_input_data_countries.query("country =='{}'".format(target_country))[rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)
        model_afolu = sm.AFOLU(sa.model_attributes)
        df_model_data_project = model_afolu.project(input_pivot)

    if models_run == "CircularEconomy":
        calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
        x_mean = df_input_data_countries.query("country =='{}'".format(target_country))[calib_targets].mean() * params
        b1 = pd.DataFrame.from_dict({j:[i]*9 for i,j in zip(x_mean,calib_targets)})
        partial_rest_parameters_df = df_input_data_countries.query("country =='{}'".format(target_country))[rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)
        model_circular_economy = CircularEconomy(sa.model_attributes)
        df_model_data_project = model_circular_economy.project(input_pivot)

    if models_run == "IPPU":
        print("\n\tRunning IPPU")
        model_ippu = sm.IPPU(sa.model_attributes)
        df_output_data = model_ippu.project(df_input_data)

    #out_vars = df_model_data_project.columns[ ["emission_co2e" in i for i in  df_model_data_project.columns]]
    out_vars = ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"]
    model_data_co2e = df_model_data_project[out_vars].sum(axis=1)
    #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
    co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))
    trend = co2_observed_data.value
    trend = [i/1000 for i in trend]

    model_data_co2e = np.array([model_data_co2e[i] for i in cv])
    trend = np.array([trend[i] for i in cv])

    output = np.mean((model_data_co2e-trend)**2)

    return output

#create fitness function
def f(X):
    return objective_a(X)

def get_mse_test(params,test):
    if models_run == "AFOLU":
        calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
        x_mean = df_input_data_countries.query("country =='{}'".format(target_country))[calib_targets].mean() * params
        b1 = pd.DataFrame.from_dict({j:[i]*9 for i,j in zip(x_mean,calib_targets)})
        partial_rest_parameters_df = df_input_data_countries.query("country =='{}'".format(target_country))[rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)
        model_afolu = sm.AFOLU(sa.model_attributes)
        df_model_data_project = model_afolu.project(input_pivot)

    if models_run == "CircularEconomy":
        calib_targets = df_calib_targets.query("model =='{}'".format(models_run))["calib_targets"]
        x_mean = df_input_data_countries.query("country =='{}'".format(target_country))[calib_targets].mean() * params
        b1 = pd.DataFrame.from_dict({j:[i]*9 for i,j in zip(x_mean,calib_targets)})
        partial_rest_parameters_df = df_input_data_countries.query("country =='{}'".format(target_country))[rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)
        model_circular_economy = CircularEconomy(sa.model_attributes)
        df_model_data_project = model_circular_economy.project(input_pivot)

    if models_run == "IPPU":
        print("\n\tRunning IPPU")
        model_ippu = sm.IPPU(sa.model_attributes)
        df_output_data = model_ippu.project(df_input_data)

    out_vars = ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"]
    model_data_co2e = df_model_data_project[out_vars].sum(axis=1)
    #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
    co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))
    trend = co2_observed_data.value
    trend = [i/1000 for i in trend]

    model_data_co2e =  np.array([model_data_co2e[i] for i in test])
    trend = np.array([trend[i] for i in test])

    output = np.mean((model_data_co2e-trend)**2)

    for i,j in zip(trend,model_data_co2e):
        print("{},{}".format(i,j))

    return output


# Define dataframe to save results
pd_output_all = pd.DataFrame()

# Define list by MPI execution
sendbuf_test = []
sendbuf_training = []
sendbuf_cv_index = []

# The process 0 will be de master. His function is create de initial matriz for Cross-Validations runs and distribute it each the process
root=0
comm = MPI.COMM_WORLD


print("TIME START CV CALIBRATION {}\n{}".format(target_country,datetime.datetime.now().time()))
if comm.rank==0:
    training,test = get_cv_index(4)
    training,test = np.array(training) , np.array(test)
    sendbuf_test = np.array_split(test,comm.size)
    sendbuf_training = np.array_split(training,comm.size)
    sendbuf_cv_index = np.array_split([i for i in range(training.shape[0])],comm.size)

test_chunk = comm.scatter(sendbuf_test,root)
training_chunk = comm.scatter(sendbuf_training,root)
index_chunk = comm.scatter(sendbuf_cv_index,root)

for k,cv,cv_test in zip(index_chunk,training_chunk,test_chunk):
    # Optimize the function
    import time
    start_time = time.time()

    print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(k,comm.rank,models_run))

    df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

    n_variables = 95
    i_sup_vec = [calib_bounds.loc[calib_bounds["variable"] == i, "max_35"].item() +0.01 for i in calib_targets]
    i_inf_vec = [calib_bounds.loc[calib_bounds["variable"] == i, "min_35"].item()  for i in calib_targets]
    precision = 8
    m = 60
    maxiter = 20
    dimension_vec = []
    genotipo = []
    length_total_cromosoma = 0

    ## Generamos población inicial
    for i in range(n_variables):
        length_cromosoma = length_variable(i_sup_vec[i],i_inf_vec[i],precision)
        length_total_cromosoma += length_cromosoma
        dimension_vec.append(length_cromosoma)
        genotipo.append(rand_population_binary(m, length_cromosoma))

    ## Iniciamos el algoritmo genético
    feno = DECODE(n_variables,m,i_sup_vec,i_inf_vec,dimension_vec,genotipo)
    print("Evaluando poblacion inicial")
    objv = OBJFUN(f,feno,False,1)

    resultados = []
    mejor_individuo = 0
    mejor_valor = 100000000

    for it in range(maxiter):
        #print(it)
        aptitud = APTITUD(objv,"min")
        seleccion = SELECCION(aptitud,"ruleta",n_variables,genotipo)
        genotipo = CRUZA(seleccion,"unpunto",length_total_cromosoma)
        genotipo = MUTACION(genotipo,length_total_cromosoma,n_variables,dimension_vec)
        feno = DECODE(n_variables,m,i_sup_vec,i_inf_vec,dimension_vec,genotipo)
        objv = OBJFUN(f,feno,False,1)
        resultados.append(min(objv))
        mejor_individuo = objv.index(min(objv))
        #print("Mejor valor fun.obj ---> {}. Variables de decision ---> {}".format(objv[mejor_individuo],feno[mejor_individuo]))
        #print("Mejor valor fun.obj ---> {}".format(objv[mejor_individuo]))

        if objv[mejor_individuo] < mejor_valor:
            mejor_valor = objv[mejor_individuo]
            mejor_vector = feno[mejor_individuo]

    print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(k,comm.rank,(time.time() - start_time)))
    print(" Cross Validation: {} on Node {}. MSE Training : {}".format(k,comm.rank,mejor_valor))
    mse_test = get_mse_test(mejor_vector,cv_test)
    print(" Cross Validation: {} on Node {}. MSE Test : {}".format(k,comm.rank,mse_test))

    pd_output = pd.DataFrame.from_dict({target:[value] for target,value in zip(calib_targets,mejor_vector)})
    pd_output["nation"] = target_country
    pd_output["MSE_training"] = mejor_valor
    pd_output["MSE_test"] = mse_test
    pd_output_all = pd.concat([pd_output_all, pd_output])

target_country = target_country.replace(" ","_")
target_country = target_country.lower()
pd_output_all.to_csv("output_calib/{}/mpi_output/{}_cv_{}_mpi_{}.csv".format(models_run,models_run,target_country,comm.rank), index = None, encoding = "UTF-8")

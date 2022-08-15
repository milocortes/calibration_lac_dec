import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel

import warnings

warnings.filterwarnings("ignore")


df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_bounds/output/calib_bounds_sector.csv")

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run)).reset_index(drop = True)

calib_bounds_groups = calib_bounds.groupby("group")
indices_params = list(calib_bounds_groups.groups[0])

for i,j in calib_bounds_groups.groups.items():
    if i!=0:
        indices_params.append(j[0])

calib_targets = calib_bounds['variable'].iloc[indices_params].reset_index(drop=True)

# Define lower and upper time bounds
year_init,year_end = 2011,2019

# Define lower and upper time bounds
t_times = range(year_init, year_end+1)

# Add variable that change
"""
countries_list = list(set(df_co2_observed_data.query("model == '{}'".format(models_run))["country"]))
countries_list.sort()
countries_list.remove("trinidad_and_tobago")
countries_list.remove("honduras")
countries_list.remove("peru")
"""
countries_list = ['argentina',
'bahamas',
'barbados',
'belize',
'brazil',
'bolivia',
'chile',
'colombia',
'costa_rica',
'dominican_republic',
'ecuador',
'el_salvador',
'guatemala',
'guyana',
'haiti',
'honduras',
'jamaica',
'mexico',
'nicaragua',
'panama',
'paraguay',
'peru',
'suriname',
'trinidad_and_tobago',
'uruguay',
'venezuela']


t_times = range(year_init, year_end+1)

path_csv = "output_calib/{}/".format(models_run)
df_afolu_all_countries = pd.DataFrame()

df_csv_all_to_save = pd.DataFrame()

for  target_country in countries_list:
    print("Colllect results {}".format(target_country))
    df_co2_observed_data_country = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))
    df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])

    target_country_ref = target_country.lower().replace(" ","_")

    df_csv_all = pd.DataFrame()
    df_csv_all_mean = pd.DataFrame()

    for i in range(8):
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
        print(target_country)
        x = df_csv_all.loc[[i]].values[0]

        calibration = CalibrationModel(df_input_country, target_country, models_run,
                                        calib_targets, calib_bounds, t_times,
                                        df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8],
                                        cv_test = [0,1,2,3,4,5,6,7,8])

        output_data = calibration.get_output_data(x)

        model_data_co2e = output_data[calibration.var_co2_emissions_by_sector["{}".format(models_run)]].sum(axis=1)

        df_afolu_partial = pd.DataFrame.from_dict({"time":[t for t in range(9)],"area":[target_country_ref]*9 ,"value": model_data_co2e,"id" : [i]*9,"type" :["simulation"]*9})
        df_afolu_country = pd.concat([df_afolu_country,df_afolu_partial])

    trend = df_co2_observed_data_country.value
    trend = [i/1000 for i in trend]

    df_afolu_historical = pd.DataFrame.from_dict({"time":[t for t in range(9)], "area":[target_country_ref]*9 ,"value": trend,"id" : [0]*9,"type" :["historical"]*9})
    df_afolu_country = pd.concat([df_afolu_country,df_afolu_historical])

    df_afolu_all_countries = pd.concat([df_afolu_all_countries,df_afolu_country])


df_csv_all_to_save_copia = df_csv_all_to_save.copy()
agrupa = calib_bounds.groupby("group")
group_list = calib_bounds["group"].unique()
total_groups = len(group_list)

for group in group_list:
    group = int(group)
    print(group)
    if group == 0:
        index_var_group = calib_bounds["variable"].iloc[agrupa.groups[group]]
        df_csv_all_to_save_copia[index_var_group] =  df_csv_all_to_save_copia[index_var_group]
    else:
        index_var_group = calib_bounds["variable"].iloc[agrupa.groups[group]]
        for col in index_var_group:
            df_csv_all_to_save_copia[col] =  df_csv_all_to_save_copia[index_var_group.iloc[0]]


df_csv_all_to_save_copia.to_csv(path_csv+"calib_vector_afolu_all_countries.csv",index = False)
df_afolu_all_countries.to_csv(path_csv+"cv_results_afolu_all_countries.csv",index = False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Set model to run
models_run = "CircularEconomy"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_CO2_data_models/output/df_calib_targets_models.csv")
calib_targets = df_calib_targets.query("model == '{}'".format(models_run))["calib_targets"]
# Load observed data
observed_by_sector = pd.read_csv("observed_data/summary_observed_by_sector.csv")
# Load all variables of sector
all_by_sector = pd.read_csv("observed_data/all_variables_CircularEconomy.csv")

list_validate_category = list(all_by_sector["var"][["frac_wali_ww_domestic_rural_treatment_path" in i or "frac_wali_ww_domestic_urban_treatment_path" in i or "frac_wali_ww_industrial_treatment_path" in i for i in all_by_sector["var"]]])

# Add to calib_targets the difference between all_by_sector and  calib_targets + observed_by_sector
excluye = list(set(list(calib_targets)+list(observed_by_sector["observed_var"]) + list_validate_category))
calib_05_150 = list(set(all_by_sector["var"]).difference(set(excluye)))

# calibration bounds
calib_bounds = pd.read_csv("bounds/model_input_variables_ce_demo.csv")
calib_bounds = calib_bounds[["variable","min_35","max_35"]]
calib_bounds = calib_bounds.query("min_35 != 1.00 and max_35 !=1.00")
calib_bounds = calib_bounds[[False if i in ['gasrf_waso_biogas','oxf_waso_average_landfilled','elasticity_protein_in_diet_to_gdppc','frac_waso_compost_methane_flared','gasrf_waso_landfill_to_ch4']  else True for i in calib_bounds["variable"]]]

calib_bounds = pd.concat([calib_bounds,pd.DataFrame.from_dict({'variable':calib_05_150 , 'min_35':[0.5]*len(calib_05_150),'max_35':[1.5]*len(calib_05_150)})])
calib_targets = calib_bounds["variable"]
# Define lower and upper time bounds
year_init,year_end = 2006,2014

t_times = range(year_init, year_end+1)

# Add variable that change

countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador',
 'el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay']

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
                                    df_co2_observed_data_country,cv_training = [0,1,2,3,4,5,6,7,8], cv_test = [0,1,2,3,4,5,6,7,8])

        output_data = calibration.get_output_data(x)

        model_data_co2e = output_data[calibration.var_co2_emissions_by_sector["CircularEconomy"]].sum(axis=1)

        df_afolu_partial = pd.DataFrame.from_dict({"time":[t for t in range(9)],"area":[target_country_ref]*9 ,"value": model_data_co2e,"id" : [i]*9,"type" :["simulation"]*9})
        df_afolu_country = pd.concat([df_afolu_country,df_afolu_partial])

    trend = df_co2_observed_data_country.value
    trend = [i/1000 for i in trend]

    df_afolu_historical = pd.DataFrame.from_dict({"time":[t for t in range(9)], "area":[target_country_ref]*9 ,"value": trend,"id" : [0]*9,"type" :["historical"]*9})
    df_afolu_country = pd.concat([df_afolu_country,df_afolu_historical])

    df_afolu_all_countries = pd.concat([df_afolu_all_countries,df_afolu_country])

df_csv_all_to_save.to_csv(path_csv+"calib_vector_circular_economy_all_countries.csv",index = False)
df_afolu_all_countries.to_csv(path_csv+"cv_results_circular_economy_all_countries.csv",index = False)

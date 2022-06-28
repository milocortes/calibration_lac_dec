import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Set model to run
models_run = "IPPU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")
df_calib_targets =  pd.read_csv("model_input_variables_ip_demo.csv")
df_calib_targets = df_calib_targets.drop_duplicates()

# Load fake_data_ippu
df_ippu = pd.read_csv("https://raw.githubusercontent.com/egobiernoytp/lac_decarbonization/main/ref/fake_data/fake_data_ippu.csv")

coincide_target_fake_ippu = list(set(df_ippu.columns).intersection(set(df_calib_targets["variable"])))

df_calib_targets = df_calib_targets[[True if i in coincide_target_fake_ippu else False for i in df_calib_targets["variable"]]]

# Load observed data
observed_by_sector = ["prodinit_ippu_chemicals_tonne",
"prodinit_ippu_plastic_tonne",
"prodinit_ippu_metals_tonne",
"prodinit_ippu_electronics_tonne",
"prodinit_ippu_textiles_tonne",
"prodinit_ippu_paper_tonne",
"prodinit_ippu_lime_and_carbonite_tonne"]


# calibration bounds
calib_bounds = df_calib_targets[["variable","min_35","max_35"]]
calib_targets = calib_bounds["variable"]
calib_targets = list(set(calib_targets).difference(observed_by_sector))
calib_bounds = calib_bounds.query("min_35 != 1.00 and max_35 !=1.00")
calib_bounds = calib_bounds[[not i if i in observed_by_sector else True for i in calib_bounds["variable"]]]
calib_bounds = calib_bounds[[not i for i in calib_bounds['min_35'].isna() |  calib_bounds['max_35'].isna()]]

calib_targets_05_1_50 = list(set(calib_targets).difference(calib_bounds["variable"]))
calib_bounds_05_1_50 = pd.DataFrame.from_dict({'variable': calib_targets_05_1_50, 'min_35':[0.5]*len(calib_targets_05_1_50),'max_35':[1.5]*len(calib_targets_05_1_50)})

calib_targets_05_1_50 += ["gdp_mmm_usd"]
calib_bounds_05_1_50 = pd.concat([calib_bounds_05_1_50, pd.DataFrame.from_dict({'variable':['gdp_mmm_usd'],'min_35':[0.005],'max_35':[2]})])

calib_bounds = pd.concat([calib_bounds,calib_bounds_05_1_50])

calib_targets = calib_bounds["variable"]

# Define lower and upper time bounds
year_init,year_end = 2006,2014

t_times = range(year_init, year_end+1)

# Add variable that change

countries_list = list(set(df_co2_observed_data.query("model == 'IPPU'")["country"]))
countries_list.sort()


t_times = range(year_init, year_end+1)

## Agregamos los vectores de calibración de CircularEconomy en clase Calubration
calib_circecon = pd.read_csv("/home/milo/Documents/egtp/LAC-dec/calibration/output_calib/CircularEconomy/calib_vector_circular_economy_all_countries.csv")
calib_circecon_calib_targets = list(calib_circecon.columns[:-2])

# Add variable that change

countries_list = ['argentina','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador','el_salvador','guatemala','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay']

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

    ## Agregamos los vectores de calibración de CircularEconomy en clase Calubration
    target_country_index = list(calib_circecon["country"]).index(target_country)
    calib_circecon_calib_vector = list(calib_circecon.iloc[target_country_index][:-2])

    for i in range(df_csv_all.shape[0]):
        print(target_country)
        x = df_csv_all.loc[[i]].values[0]

        calibration = CalibrationModel(df_input_country, target_country, models_run,
                                        calib_targets, calib_bounds, t_times,
                                        df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8],
                                        cv_test = [0,1,2,3,4,5,6,7,8],
                                        downstream = True)

        calibration.set_calib_targets('CircularEconomy',calib_circecon_calib_targets)
        calibration.set_best_vector('CircularEconomy',calib_circecon_calib_vector)

        output_data = calibration.get_output_data(x)

        model_data_co2e = output_data[calibration.var_co2_emissions_by_sector["IPPU"]].sum(axis=1)

        df_afolu_partial = pd.DataFrame.from_dict({"time":[t for t in range(9)],"area":[target_country_ref]*9 ,"value": model_data_co2e,"id" : [i]*9,"type" :["simulation"]*9})
        df_afolu_country = pd.concat([df_afolu_country,df_afolu_partial])

    trend = df_co2_observed_data_country.value
    trend = [i/1000 for i in trend]

    df_afolu_historical = pd.DataFrame.from_dict({"time":[t for t in range(9)], "area":[target_country_ref]*9 ,"value": trend,"id" : [0]*9,"type" :["historical"]*9})
    df_afolu_country = pd.concat([df_afolu_country,df_afolu_historical])

    df_afolu_all_countries = pd.concat([df_afolu_all_countries,df_afolu_country])

df_csv_all_to_save.to_csv(path_csv+"calib_vector_ippu_all_countries.csv",index = False)
df_afolu_all_countries.to_csv(path_csv+"cv_results_ippu_all_countries.csv",index = False)

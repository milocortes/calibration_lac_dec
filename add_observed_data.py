import numpy as np
import pandas as pd

import glob

import statsmodels.api as statsm

# Load input data: fake_data_complete.csv
df_input_data = pd.read_csv("lac_decarbonization/ref/fake_data/fake_data_complete.csv")

# All params
all_params = df_input_data.columns

'''
Add observed data to df_input_data
'''
# For each country make a copy of df_input_data
all_countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador',
 'el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']

df_input_data_countries = pd.DataFrame()

for country in all_countries_list:
    partial = df_input_data.iloc[0:6]
    partial["country"] = country
    df_input_data_countries = pd.concat([df_input_data_countries,partial])

# Load csv for specific model
sectores = ["all_sectors","AFOLU","CircularEconomy","IPPU"]

list_csv_obs_data = []

for sector in sectores:
    list_csv_obs_data +=  [i.split("/")[-1][:-4] for i in glob.glob("observed_data/{}/*.csv".format(sector))]

# Recollect observed data
dic_validate_category = {"frac_wali_ww_domestic_rural_treatment_path" : list(all_params[["frac_wali_ww_domestic_rural_treatment_path" in i for i in all_params]]),
                        "frac_wali_ww_domestic_urban_treatment_path" : list(all_params[["frac_wali_ww_domestic_urban_treatment_path" in i for i in all_params]]),
                        "frac_agrc_initial_yield":list(all_params[["frac_agrc_initial_yield" in i for i in all_params]]),
                        "frac_agrc_initial_area_cropland" :list(all_params[["frac_agrc_initial_area_cropland" in i for i in all_params]])}

#for i in list_csv_obs_data:
resumen_observadas = pd.read_csv("observed_data/resumen_variables_observadas.csv")
resumen_observadas.query("in_inputs==1", inplace = True)
observed_data_change_over_time = resumen_observadas.query("var_change_time ==1")["variable"].to_list()
observed_data_constant = list(set(list_csv_obs_data) -set(observed_data_change_over_time))
observed_data_constant_intersect = list(set(all_params).intersection(observed_data_constant))
observed_data_constant_intersect.sort()

dict_resumen_observadas = {i:j for i,j in zip(resumen_observadas.variable,resumen_observadas.sector)}

print("\nAdding Variables constant over time\n")

for i in observed_data_constant_intersect:
    print("Adding {} data".format(i))
    #df_csv_obs_data = pd.read_csv("observed_data/{}/{}.csv".format(models_run,i))
    df_csv_obs_data = pd.read_csv("observed_data/{}/{}.csv".format(dict_resumen_observadas[i],i))
    df_csv_obs_data["Nation"] = df_csv_obs_data["Nation"].apply(lambda x: x.lower().replace(" ","_").replace("_(bolivarian_republic_of)","").replace("_(plurinational_state_of)",""))
    country_in_observed_data = set(list(df_csv_obs_data["Nation"]))
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
            #df_input_data_countries[i][df_input_data_countries['country'] == country] = mean_value_obs
            df_input_data_countries.loc[df_input_data_countries['country']==country,i] = mean_value_obs


# Verify group values restrictions
print("\nVerify group values restrictions\n")
for category,variables in dic_validate_category.items():
    print("Check group value restrictions {}".format(category))
    for country in all_countries_list:
        country_partial_df_input_data_countries = df_input_data_countries.query("country =='{}'".format(country))
        total = sum(country_partial_df_input_data_countries[variables].loc[0])
        if total > 0:
            for var in variables:
                value_var = df_input_data_countries.loc[df_input_data_countries["country"]==country,var]
                df_input_data_countries.loc[df_input_data_countries["country"]==country,var] = value_var/total

# Add variable that change

print("\nAdding Variables change over time\n")

year_init,year_end = 2014,2019

for var in observed_data_change_over_time:
    print("Adding {}".format(var))
    df_var_change = pd.read_csv("observed_data/{}/{}.csv".format(dict_resumen_observadas[var],var))
    if var != "gdp_mmm_usd":
        for country in all_countries_list:
            df_var_change["Nation"] = df_var_change["Nation"].apply(lambda x: x.lower().replace(" ","_").replace("_(bolivarian_republic_of)","").replace("_(plurinational_state_of)",""))
            if country in set(df_var_change["Nation"]):
                df_var_change_country = df_var_change.query("Nation =='{}' and (Year >= {} and Year <= {})".format(country,year_init,year_end))
                if df_var_change_country.shape[0] >=(year_end-year_init+1):
                    if df_var_change_country.shape[0] > (year_end-year_init+1):
                        df_input_data_countries.loc[df_input_data_countries["country"]==country, var] = df_var_change_country.groupby(["Nation","Year"]).sum()[var].to_numpy()
                    else:
                        df_input_data_countries.loc[df_input_data_countries["country"]==country, var] = df_var_change_country[var].to_numpy()
    else:
        for country in all_countries_list:
            if country != "venezuela":
                df_var_change_country = df_var_change.query("Nation =='{}' and (Year >= {} and Year <= {})".format(country,year_init,year_end))
                cycle, trend = statsm.tsa.filters.hpfilter(df_var_change_country[var].to_numpy(), 1600)
                df_input_data_countries.loc[df_input_data_countries["country"]==country, var] = trend



croplands = []
for i in df_input_data_countries.columns:
    if "frac_agrc_initial_area_cropland" in i:
        croplands.append(i)

df_input_data_countries[croplands] = df_input_data_countries[croplands].div(df_input_data_countries[croplands].sum(axis=1),axis=0)

df_input_data_countries.to_csv("all_countries_test_CalibrationModel_class.csv", index = False)

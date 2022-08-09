import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel,Socioeconomic,sa

import warnings

warnings.filterwarnings("ignore")

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Define target country
target_country = "brazil"

# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_bounds/output/calib_bounds_sector.csv")

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets = calib_bounds['variable']

# Define lower and upper time bounds
year_init,year_end = 2011,2019

df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))

calibration = CalibrationModel(df_input_country, target_country, models_run,
                                calib_targets, calib_bounds, t_times,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8] , 
                                cv_test = [1,3,5,7], cv_calibration = False)

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item()))  for i in calibration.calib_targets["AFOLU"]]

calibration.f(X)
calibration.run_calibration("differential_evolution", population = 40, maxiter = 20)
plt.plot(calibration.fitness_values["AFOLU"])
plt.show()


output_data = calibration.get_output_data(calibration.best_vector["AFOLU"])
co2_computed = output_data[calibration.var_co2_emissions_by_sector["AFOLU"]].sum(axis=1)
plt.plot([i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(target_country)
plt.legend()
plt.show()


#### TEST BOUND OPTIMIZATION


start_time = time.time()


def min_bounds(X):
    calib_bounds_genetic_co2 = calib_bounds.query("weight_co2==1")
    calib_bounds_genetic_vars = calib_bounds.query("weight_co2==0")


print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))


n_variables = len(self.calib_targets[self.subsector_model])
i_sup_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]]
i_inf_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]
precision = 8

binary_genetic = BinaryGenetic(population,n_variables,i_sup_vec,i_inf_vec,precision,maxiter)
self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor= binary_genetic.run_optimization(self.f)

print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))



all_countries = ['argentina',
'bahamas',
'barbados',
'belize',
'bolivia',
'brazil',
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

df_acumula = pd.DataFrame()

for target_country in all_countries:
    print(target_country)
    df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
    model_socio = Socioeconomic(sa.model_attributes)
    df_project, df_growth_rates = model_socio.project(df_input_country)
    df_parcial = pd.DataFrame({"Nation" : [target_country]*9, "Year" :range(2011,2020),"qty_gnrl_households" : df_project["qty_gnrl_households"].to_list()})
    df_acumula = pd.concat([df_acumula,df_parcial])
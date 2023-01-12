import os

os.environ['LAC_PATH'] = '/home/milo/Documents/julia/lac_decarbonization'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sisepuede_calibration.calibration_lac import CalibrationModel

data_path = "/home/milo/Documents/egap/SISEPUEDE/packaging_projects/github_projects/sisepuede_calibration_dev/unit_testing/data_test"

#df_input_all_countries = pd.read_csv("/home/milo/Documents/egtp/LAC-dec/verifica/calibration_lac_dec-30092022/calibration_lac_dec/real_data_2022_10_04.csv")
df_input_all_countries = pd.read_csv( os.path.join(data_path, "data_complete_future_2022_12_08_test.csv") )
df_input_all_countries.rename(columns = {"area_country_ha":"area_gnrl_country_ha"}, inplace = True)

faltan = ['mcf_trww_treated_secondary_anaerobic', 'mcf_trww_treated_advanced_anaerobic', 'frac_trww_tow_removed_treated_secondary_anaerobic', 'gasrf_trww_biogas_treated_advanced_anaerobic', 'physparam_krem_sludge_factor_treated_advanced_aerobic', 'frac_trww_n_removed_treated_latrine_improved', 'ef_trww_treated_advanced_anaerobic_g_n2o_per_g_n', 'frac_waso_compost_sludge', 'ef_trww_treated_secondary_anaerobic_g_n2o_per_g_n', 'gasrf_waso_biogas', 'ef_trww_treated_secondary_aerobic_g_n2o_per_g_n', 'frac_waso_biogas_sludge', 'physparam_wali_p_per_cod', 'frac_trww_n_removed_treated_advanced_anaerobic', 'gasrf_trww_biogas_treated_secondary_aerobic', 'frac_trww_n_removed_treated_secondary_anaerobic', 'frac_trww_tow_removed_treated_advanced_aerobic', 'mcf_trww_treated_advanced_aerobic', 'frac_trww_tow_removed_treated_primary', 'ef_trww_treated_advanced_aerobic_g_n2o_per_g_n', 'frac_trww_tow_removed_treated_secondary_aerobic', 'mcf_trww_treated_secondary_aerobic', 'ef_waso_compost_kg_ch4_per_kg_sludge', 'gasrf_trww_biogas_treated_secondary_anaerobic', 'frac_trww_n_removed_treated_advanced_aerobic', 'frac_trww_n_removed_treated_septic', 'gasrf_trww_biogas_treated_advanced_aerobic', 'frac_gnrl_eating_red_meat', 'ef_trww_treated_primary_g_n2o_per_g_n', 'ef_waso_biogas_kg_ch4_per_kg_sludge', 'ef_waso_compost_kg_n2o_per_kg_sludge', 'mcf_trww_treated_primary', 'frac_trww_n_removed_treated_primary', 'occrateinit_gnrl_occupancy', 'frac_trww_p_removed_treated_advanced_anaerobic', 'frac_trww_n_removed_treated_latrine_unimproved', 'frac_trww_tow_removed_treated_advanced_anaerobic', 'frac_trww_p_removed_treated_advanced_aerobic', 'physparam_wali_p_per_bod', 'physparam_krem_sludge_factor_treated_secondary_aerobic', 'elasticity_gnrl_rate_occupancy_to_gdppc', 'frac_trww_n_removed_treated_secondary_aerobic','limit_gnrl_annual_emissions_mt_ch4', 'limit_gnrl_annual_emissions_mt_co2', 'limit_gnrl_annual_emissions_mt_n2o']
faltan += ['frac_wali_ww_industrial_treatment_path_primary', 'frac_wali_ww_industrial_treatment_path_advanced_anaerobic', 'frac_wali_ww_domestic_urban_treatment_path_advanced_anaerobic', 'frac_wali_ww_domestic_urban_treatment_path_primary', 'frac_wali_ww_domestic_urban_treatment_path_advanced_aerobic', 'frac_trww_p_removed_untreated_no_sewerage', 'frac_trww_p_removed_treated_secondary_anaerobic', 'frac_trww_p_removed_treated_primary', 'frac_wali_ww_domestic_urban_treatment_path_secondary_aerobic', 'frac_trww_p_removed_treated_latrine_unimproved', 'frac_trww_p_removed_treated_secondary_aerobic', 'frac_wali_ww_domestic_rural_treatment_path_advanced_anaerobic', 'frac_wali_ww_domestic_rural_treatment_path_advanced_aerobic', 'frac_wali_ww_industrial_treatment_path_secondary_aerobic', 'frac_wali_ww_domestic_rural_treatment_path_primary', 'frac_wali_ww_industrial_treatment_path_advanced_aerobic', 'frac_trww_p_removed_untreated_with_sewerage', 'frac_trww_p_removed_treated_septic', 'frac_wali_ww_domestic_urban_treatment_path_secondary_anaerobic', 'frac_wali_ww_domestic_rural_treatment_path_secondary_anaerobic', 'frac_wali_ww_industrial_treatment_path_secondary_anaerobic', 'frac_trww_p_removed_treated_latrine_improved', 'frac_trww_n_removed_untreated_no_sewerage', 'frac_trww_n_removed_untreated_with_sewerage', 'frac_wali_ww_domestic_rural_treatment_path_secondary_aerobic']
fake_data = pd.read_csv("https://raw.githubusercontent.com/egobiernoytp/lac_decarbonization/main/ref/fake_data/fake_data_complete.csv")

# Define target country
target_country = "Turkey"

# Set model to run
models_run = "CircularEconomy"

# Load observed CO2 data
#df_co2_observed_data = pd.read_csv("/home/milo/Documents/egtp/LAC-dec/verifica/calibration_lac_dec-30092022/calibration_lac_dec/build_CO2_data_models/output/co2_all_models.csv")
df_co2_observed_data = pd.read_csv( os.path.join(data_path, "turkey_co2.csv") )

# Load calib targets by model to run
df_calib_targets =  pd.read_csv( os.path.join(data_path, "calib_bounds_sector.csv"))

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets = calib_bounds['variable']

# Define lower and upper time bounds
year_init,year_end = 2011,2019

df_input_country = df_input_all_countries.query("nation =='{}'".format(target_country)).reset_index().drop(columns=["index"])
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))
df_input_country_all_time_period = df_input_all_countries.query("nation =='{}'".format(target_country)).reset_index().drop(columns=["index"])

for f in faltan:
    df_input_country[f] = fake_data[f]
    df_input_country_all_time_period[f] = fake_data[f]

# Instance of CalibrationModel
calibration = CalibrationModel(year_init, year_end, df_input_country, target_country, models_run,
                                calib_targets, calib_bounds,df_input_country_all_time_period,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8] ,cv_calibration = False,precition=4)
# Test function evaluation
X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item() +0.01))  for i in calibration.calib_targets["CircularEconomy"]]
calibration.f(X)


# Setup and run calibration with PSO

parameters_algo = {"alpha" : 0.5, "beta" : 0.8}

calibration.run_calibration("pso", population = 20, maxiter = 10, param_algo = parameters_algo)

# Check fitness
plt.plot(calibration.fitness_values["CircularEconomy"])
plt.show()

# Check performance

output_data = calibration.get_output_data(calibration.best_vector["CircularEconomy"])
co2_computed = output_data[calibration.var_co2_emissions_by_sector["CircularEconomy"]].sum(axis=1)
plt.plot(range(year_init,year_end+1),[i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(range(year_init,year_end+1),co2_computed,label="estimado")
plt.title("Turkey")
plt.legend()
plt.show()

import pandas as pd

'''
# Build CircularEconomy target and bounds
'''

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("../data/df_calib_targets_models.csv")
calib_targets = df_calib_targets.query("model == '{}'".format("CircularEconomy"))["calib_targets"]
# Load observed data
observed_by_sector = pd.read_csv("../data/summary_observed_by_sector.csv")
# Load all variables of sector
all_by_sector = pd.read_csv("../data/all_variables_CircularEconomy.csv")

list_validate_category = list(all_by_sector["var"][["frac_wali_ww_domestic_rural_treatment_path" in i or "frac_wali_ww_domestic_urban_treatment_path" in i or "frac_wali_ww_industrial_treatment_path" in i for i in all_by_sector["var"]]])

# Add to calib_targets the difference between all_by_sector and  calib_targets + observed_by_sector
excluye = list(set(list(calib_targets)+list(observed_by_sector["observed_var"]) + list_validate_category))
calib_05_150 = list(set(all_by_sector["var"]).difference(set(excluye)))

# calibration bounds
calib_bounds = pd.read_csv("../data/model_input_variables_ce_demo.csv")
calib_bounds = calib_bounds[["variable","min_35","max_35"]]
calib_bounds = calib_bounds.query("min_35 != 1.00 and max_35 !=1.00")
calib_bounds = calib_bounds[[False if i in ['gasrf_waso_biogas','oxf_waso_average_landfilled','elasticity_protein_in_diet_to_gdppc','frac_waso_compost_methane_flared','gasrf_waso_landfill_to_ch4']  else True for i in calib_bounds["variable"]]]

calib_bounds = pd.concat([calib_bounds,pd.DataFrame.from_dict({'variable':calib_05_150 , 'min_35':[0.5]*len(calib_05_150),'max_35':[1.5]*len(calib_05_150)})])

calib_bounds_ce = calib_bounds
calib_bounds_ce['sector'] = "CircularEconomy"
calib_bounds_ce['var_change_over_time'] = 0

'''
# Build IPPU target and bounds
'''

# Load calib targets 
df_calib_targets =  pd.read_csv("../data/model_input_variables_ip_demo.csv")
df_calib_targets = df_calib_targets.drop_duplicates()

# Load fake_data_ippu
df_ippu = pd.read_csv("../../lac_decarbonization/ref/fake_data/fake_data_ippu.csv")

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

calib_bounds_ippu = calib_bounds
calib_bounds_ippu['sector'] = 'IPPU'
calib_bounds_ippu['var_change_over_time'] = [0]*85 + [1]

calib_bounds_sectors = pd.concat([calib_bounds_ce,calib_bounds_ippu])

calib_bounds_sectors.to_csv("../output/calib_bounds_sector.csv")
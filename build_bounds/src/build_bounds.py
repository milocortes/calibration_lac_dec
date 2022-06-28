import pandas as pd
import numpy as np



'''
Build AFOLU target and bounds
'''
df_afolu_calib_targets = pd.read_csv("../../lac_decarbonization/ref/preliminary_calibration_info/df_afolu_var_calib.csv")
df_afolu_calib_targets = df_afolu_calib_targets[['variable','min_35','max_35']]
'''
Build CircularEconomy target and bounds
'''






#df_waste_calib_targets = pd.read_csv("df_waste_var_calib.csv")
df_waste_calib_targets = pd.read_excel("df_ce_var_calib.xlsx", skiprows=[0,1])
df_waste_calib_targets = df_waste_calib_targets.query("Calibration==1")
#waste_calib_targets = df_waste_calib_targets["waste_var_calib"]
waste_calib_targets = df_waste_calib_targets["Variable"]
#waste_var_in_01_interval = waste_calib_targets[df_waste_calib_targets["var_in_01_interval"].fillna(0).apply(lambda x: bool(x))]

# AFOLU calib targets
df_afolu_calib_targets = pd.read_csv("afolu_input_template_with_calib_js.csv")
afolu_calib_targets =  df_afolu_calib_targets.query("calib==1")["variable"]

# Labels AFOLU and CircularEconomy targets
labels_models = ["AFOLU" for i in afolu_calib_targets] + ["CircularEconomy" for i in waste_calib_targets]

#var_in_01_interval = [0 for i in range(len(afolu_calib_targets))] + list(df_waste_calib_targets["var_in_01_interval"].fillna(0).apply(lambda x: int(x)))
var_in_01_interval = [0 for i in range(len(afolu_calib_targets))] + list(df_waste_calib_targets["Variable in [0, 1] Interval"].fillna(0).apply(lambda x: int(x)))

df_calib_targets_models = pd.DataFrame.from_dict({"model":labels_models,"calib_targets":list(afolu_calib_targets)+list(waste_calib_targets), "var_in_01_interval": var_in_01_interval})


#df_waste_calib_targets = pd.read_csv("df_waste_var_calib.csv")
df_waste_calib_targets = pd.read_excel("../data/df_ce_var_calib.xlsx", skiprows=[0,1])
df_waste_calib_targets = df_waste_calib_targets.query("Calibration==1")
#waste_calib_targets = df_waste_calib_targets["waste_var_calib"]
waste_calib_targets = df_waste_calib_targets["Variable"]
#waste_var_in_01_interval = waste_calib_targets[df_waste_calib_targets["var_in_01_interval"].fillna(0).apply(lambda x: bool(x))]

# AFOLU calib targets
df_afolu_calib_targets = pd.read_csv("../data/afolu_input_template_with_calib_js.csv")
afolu_calib_targets =  df_afolu_calib_targets.query("calib==1")["variable"]

# Labels AFOLU and CircularEconomy targets
labels_models = ["AFOLU" for i in afolu_calib_targets] + ["CircularEconomy" for i in waste_calib_targets]

#var_in_01_interval = [0 for i in range(len(afolu_calib_targets))] + list(df_waste_calib_targets["var_in_01_interval"].fillna(0).apply(lambda x: int(x)))
var_in_01_interval = [0 for i in range(len(afolu_calib_targets))] + list(df_waste_calib_targets["Variable in [0, 1] Interval"].fillna(0).apply(lambda x: int(x)))

df_calib_targets_models = pd.DataFrame.from_dict({"model":labels_models,"calib_targets":list(afolu_calib_targets)+list(waste_calib_targets), "var_in_01_interval": var_in_01_interval})


# Load calib targets by model to run
df_calib_targets =  pd.read_csv("../data/df_calib_targets_models.csv")
calib_targets = df_calib_targets.query("model == '{}'".format(models_run))["calib_targets"]
# Load observed data
observed_by_sector = pd.read_csv("../data/summary_observed_by_sector.csv")
# Load all variables of sector
all_by_sector = pd.read_csv("../data/all_variables_CircularEconomy.csv")

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
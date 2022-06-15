import pandas as pd
import os

# Change dir
os.chdir("../data")

# Waste calib targets
'''
waste_calib_targets = ["benefit_per_capita_no_red_meat","carrying_capacity_scalar","density_wali_kg_n_per_m3_ww_industrial","ef_anaerobicdom","ef_biomassburning","ef_biomassdom",
"ef_boc_wetlands_gg_ch4_ha","ef_forestfires","ef_forestmethane","ef_lnduconv_croplands","ef_lnduconv_forests_mangroves","ef_lnduconv_forests_primary","ef_lnduconv_forests_secondary","ef_lnduconv_grasslands","ef_lnduconv_other","ef_lnduconv_settlements",
"ef_lnduconv_settlements","ef_lnduconv_wetlands","ef_lvst_entferm","ef_lvst_mm","ef_pasturelimfrt_grasslands_gg_n2o_ha","ef_sequestration","ef_soil_carbon_grasslands_gg_co2_ha",
"ef_soilcarbon","frac_area_cropland_calculated_","lvst_pop_"]
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

# Change dir and save targets
os.chdir("../output")
df_calib_targets_models.to_csv("df_calib_targets_models.csv", index = False)

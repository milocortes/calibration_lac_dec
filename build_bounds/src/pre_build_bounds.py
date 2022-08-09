import pandas as pd
import numpy as np
import yaml
'''
Read observed data
'''

observed_data = pd.read_csv("../../observed_data/resumen_variables_observadas.csv")

'''
Read CO2 sector weights
'''
with open(r'../data/weights_co2_sectors.yaml') as file:
    weights_co2_sectors = yaml.load(file, Loader=yaml.FullLoader)


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
#calib_bounds_ce['weight_co2'] = 0

'''
# Build IPPU target and bounds
'''
## ef_ippu_tonne
ef_ippu_tonne_list = ['ef_ippu_tonne_c2f6_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_c2f6_per_tonne_production_electronics',
                        'ef_ippu_tonne_c2h3f3_per_mmm_gdp_product_use_ods_refrigeration',
                        'ef_ippu_tonne_c2hf5_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_c2hf5_per_mmm_gdp_product_use_ods_refrigeration',
                        'ef_ippu_tonne_c4h5f5_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_c5h2f10_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_c6f14_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_cc4f8_per_tonne_production_electronics',
                        'ef_ippu_tonne_cf4_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_cf4_per_tonne_production_electronics',
                        'ef_ippu_tonne_ch2f2_per_mmm_gdp_product_use_ods_refrigeration',
                        'ef_ippu_tonne_ch2f2_per_tonne_production_electronics',
                        'ef_ippu_tonne_ch3chf2_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_ch3chf2_per_mmm_gdp_product_use_ods_refrigeration',
                        'ef_ippu_tonne_ch4_per_tonne_production_chemicals',
                        'ef_ippu_tonne_ch4_per_tonne_production_metals',
                        'ef_ippu_tonne_ch4_per_tonne_production_plastic',
                        'ef_ippu_tonne_chf2cf3_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_chf2cf3_per_mmm_gdp_product_use_ods_refrigeration',
                        'ef_ippu_tonne_chf3_per_mmm_gdp_product_use_ods_other',
                        'ef_ippu_tonne_chf3_per_mmm_gdp_product_use_ods_refrigeration']


ef_ippu_tonne_list = list(set(ef_ippu_tonne_list))

calib_ef_ippu_tonne = pd.DataFrame()

for i in ef_ippu_tonne_list:
    parcial_calib_ef_ippu_tonne = pd.DataFrame.from_dict({'variable':[i],'min_35':[0.00000001],'max_35':[ 0.0003]})
    calib_ef_ippu_tonne = pd.concat([calib_ef_ippu_tonne,parcial_calib_ef_ippu_tonne])


# Load calib targets
df_calib_targets =  pd.read_csv("../data/model_input_variables_ip_demo.csv")
df_calib_targets = df_calib_targets.drop_duplicates()

# Load fake_data_ippu
df_ippu = pd.read_csv("../../lac_decarbonization/ref/fake_data/fake_data_ippu.csv")

coincide_target_fake_ippu = list(set(df_ippu.columns).intersection(set(df_calib_targets["variable"])))

df_calib_targets = df_calib_targets[[True if i in coincide_target_fake_ippu else False for i in df_calib_targets["variable"]]]

descarta_ef_ippu_tonne = list(set(df_calib_targets["variable"]).intersection(ef_ippu_tonne_list))
df_calib_targets = df_calib_targets[[not i in descarta_ef_ippu_tonne for i in df_calib_targets["variable"]]]


# Load observed data
observed_by_sector = ["prodinit_ippu_cement_tonne",
                        "prodinit_ippu_chemicals_tonne",
                        "prodinit_ippu_electronics_tonne",
                        "prodinit_ippu_glass_tonne",
                        "prodinit_ippu_lime_and_carbonite_tonne",
                        "prodinit_ippu_metals_tonne",
                        "prodinit_ippu_paper_tonne",
                        "prodinit_ippu_plastic_tonne",
                        "prodinit_ippu_rubber_and_leather_tonne",
                        "prodinit_ippu_textiles_tonne"]


# calibration bounds
calib_bounds = df_calib_targets[["variable","min_35","max_35"]]
calib_targets = calib_bounds["variable"]
calib_targets = list(set(calib_targets).difference(observed_by_sector))
calib_bounds = calib_bounds.query("min_35 != 1.00 and max_35 !=1.00")
calib_bounds = calib_bounds[[not i if i in observed_by_sector else True for i in calib_bounds["variable"]]]
calib_bounds = calib_bounds[[not i for i in calib_bounds['min_35'].isna() |  calib_bounds['max_35'].isna()]]

calib_targets_05_1_50 = list(set(calib_targets).difference(calib_bounds["variable"]))
calib_bounds_05_1_50 = pd.DataFrame.from_dict({'variable': calib_targets_05_1_50, 'min_35':[0.5]*len(calib_targets_05_1_50),'max_35':[1.5]*len(calib_targets_05_1_50)})

descarta_ef_ippu_tonne = list(set(calib_bounds_05_1_50["variable"]).intersection(ef_ippu_tonne_list))

calib_bounds_05_1_50 = calib_bounds_05_1_50[[not i in descarta_ef_ippu_tonne for i in calib_bounds_05_1_50["variable"]]]

calib_targets_05_1_50 += ["gdp_mmm_usd"]
calib_bounds_05_1_50 = pd.concat([calib_bounds_05_1_50, pd.DataFrame.from_dict({'variable':['gdp_mmm_usd'],'min_35':[0.005],'max_35':[2]})])

calib_bounds = pd.concat([calib_bounds,calib_ef_ippu_tonne,calib_bounds_05_1_50])

calib_bounds_ippu = calib_bounds
calib_bounds_ippu['sector'] = 'IPPU'
calib_bounds_ippu['var_change_over_time'] = [0]*(calib_bounds_ippu.shape[0]-1) + [1]

#calib_bounds_ippu['weight_co2'] = 0
#weights_co2_ippu = pd.DataFrame([(i,0,1, "IPPU", 0,1) for i in weights_co2_sectors["IPPU"]],columns=calib_bounds_ippu.columns)
#calib_bounds_ippu = pd.concat([calib_bounds_ippu,weights_co2_ippu])


'''
# Build AFOLU target and bounds
'''

fake_afolu = pd.read_csv("../../lac_decarbonization/ref/fake_data/fake_data_afolu.csv")
observed_data_afolu = observed_data.query("sector == 'AFOLU'")["variable"].to_list()
descarta = ['time_period','area_gnrl_country_ha','gdp_mmm_usd','va_commercial_mmm_usd','va_industrial_mmm_usd','va_manufacturing_mmm_usd','va_mining_mmm_usd','population_gnrl_rural','population_gnrl_urban']
descarta +=['frac_agrc_initial_area_cropland_bevs_and_spices',
 'frac_agrc_initial_area_cropland_cereals',
 'frac_agrc_initial_area_cropland_fibers',
 'frac_agrc_initial_area_cropland_fruits',
 'frac_agrc_initial_area_cropland_herbs_and_other_perennial_crops',
 'frac_agrc_initial_area_cropland_nuts',
 'frac_agrc_initial_area_cropland_other_annual',
 'frac_agrc_initial_area_cropland_other_woody_perennial',
 'frac_agrc_initial_area_cropland_pulses',
 'frac_agrc_initial_area_cropland_rice',
 'frac_agrc_initial_area_cropland_sugar_cane',
 'frac_agrc_initial_area_cropland_tubers',
 'frac_agrc_initial_area_cropland_vegetables_and_vines'] + observed_data_afolu
 

afolu_calib_target = list(fake_afolu.columns[[not i in descarta for i in fake_afolu.columns]])
calib_bounds_afolu = pd.DataFrame.from_dict({'variable' : afolu_calib_target ,
                                             'min_35' : [0.01] * len(afolu_calib_target) ,
                                             'max_35' : [2.0] * len(afolu_calib_target) ,
                                             'sector' : ['AFOLU']*len(afolu_calib_target),
                                             'var_change_over_time' : [0] * len(afolu_calib_target) })


prefix_scalar_between_0_1 = ['frac_agrc_initial_area_cropland','frac_agrc_initial_yield_feed','frac_gnrl_eating_red_meat','frac_lndu_initial_','pij_lndu_croplands_to',
'pij_lndu_forests_mangroves_to','pij_lndu_forests_primary_to','pij_lndu_forests_secondary_to','pij_lndu_grasslands_to','pij_lndu_other_to_','pij_lndu_settlements_to','pij_lndu_wetlands_to','scalar_lvst_carrying_capacity']

var_scalar_between_0_1 = []

for prefijo in prefix_scalar_between_0_1:
    afolu_calib_target_np = np.array(afolu_calib_target)
    var_scalar_between_0_1 += [i for i in afolu_calib_target if i.startswith(prefijo)]

var_scalar_between_0_1 = [str(i) for i in var_scalar_between_0_1]
calib_bounds_afolu.loc[calib_bounds_afolu["variable"].isin(var_scalar_between_0_1) ,'max_35'] = 1

#calib_bounds_afolu['weight_co2'] = 0
#weights_co2_afolu = pd.DataFrame([(i,0,1, "AFOLU", 0,1) for i in weights_co2_sectors["AFOLU"]],columns=calib_bounds_ippu.columns)
#calib_bounds_afolu = pd.concat([calib_bounds_afolu,weights_co2_afolu])

predefined_afolu_vars = pd.read_csv("../data/predefined_data_vars.csv")
predefined_afolu_vars = list(set(predefined_afolu_vars["file"]).difference(afolu_calib_target))

calib_bounds_afolu_predefined = pd.DataFrame.from_dict({'variable' : predefined_afolu_vars ,
                                             'min_35' : [0.5] * len(predefined_afolu_vars) ,
                                             'max_35' : [1.5] * len(predefined_afolu_vars) ,
                                             'sector' : ['AFOLU']*len(predefined_afolu_vars),
                                             'var_change_over_time' : [1] * len(predefined_afolu_vars) })

calib_bounds_afolu = pd.concat([calib_bounds_afolu,calib_bounds_afolu_predefined])

'''
Concat all sectors
'''
calib_bounds_sectors = pd.concat([calib_bounds_ce,calib_bounds_ippu,calib_bounds_afolu])

calib_bounds_sectors.to_csv("../output/calib_bounds_sector.csv",index=False)

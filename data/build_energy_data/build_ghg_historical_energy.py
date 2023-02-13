import pandas as pd

# Load emission data from climate watch 
ghg_energy = pd.read_csv("historical_emissions.csv")

# Get specific columns
ghg_energy = ghg_energy[["Country", "Sector", "Gas"] + [str(i) for i in range(1990, 2020)]] 

# From short to long
ghg_energy = pd.melt(ghg_energy, id_vars= ["Country", "Sector", "Gas"])

# Replace NaN by 0
ghg_energy = ghg_energy.fillna(0)
ghg_energy["key"] = ghg_energy["Sector"] + "-" + ghg_energy["Gas"]

# Load data crosswalks sisepuede-climate_watch
crosswalk = pd.read_excel("sisepuede_fields_to_climate_watch_emissions_map.xlsx")
crosswalk = crosswalk[crosswalk.columns[:5]].query("include ==1").reset_index(drop = True)
crosswalk["gas"] = crosswalk["gas"].str.upper()

# load faltantes
#faltantes = pd.read_csv("faltan_energia.csv")

# remove faltantes
#crosswalk = crosswalk[~crosswalk["fields_to_sum"].isin(faltantes["faltan"])].reset_index(drop = True)

# Group by subsector sisepuede
grouped_crosswalk = crosswalk.groupby("subsector_sisepuede") 

ghg_energy["subsector_sisepuede"] = "subsector"

for subsector in grouped_crosswalk.groups.keys():
    sector_gas = {f"{i}-{j}" for i,j in zip(grouped_crosswalk.get_group(subsector)["category_climate_watch"], grouped_crosswalk.get_group(subsector)["gas"])} 
    ghg_energy.loc[ghg_energy["key"].isin(sector_gas),"subsector_sisepuede"] = subsector

ghg_energy = ghg_energy.query("subsector_sisepuede != 'subsector'") 

ghg_energy = ghg_energy[["Country", "subsector_sisepuede", "variable", "value"]]

ghg_energy = ghg_energy.groupby(["Country", "subsector_sisepuede", "variable"]).sum().reset_index()

# Load ISO3 code
iso3_m49_correspondence = pd.read_html("https://unstats.un.org/unsd/methodology/m49/")[0]

iso3_m49_correspondence.rename(columns = {"Country or Area" : "Country", "ISO-alpha3 code" : "iso_code3"}, inplace = True)
iso3_m49_correspondence = iso3_m49_correspondence[["Country", "iso_code3"]]

ghg_energy = ghg_energy.merge(right=iso3_m49_correspondence, how = 'inner', on = 'Country')
ghg_energy.to_csv("ghg_LAC_energy_iso_code3.csv", index = False)

# Correspondence subsector-columns sisepuede
subsector_fiels_to_sum = {}

for subsector, indices in grouped_crosswalk.groups.items():
    subsector_fiels_to_sum[subsector] = crosswalk.loc[indices, "fields_to_sum"].to_list()

import json

json.dump(subsector_fiels_to_sum, open("../energy_subsector_items.json", "w")) 
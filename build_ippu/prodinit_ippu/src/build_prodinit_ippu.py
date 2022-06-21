from fredapi import Fred
import pandas as pd
import yaml
import math

# Set credentials
fred = Fred(api_key='51a7c360eef6807d0b4b513d4cce059c')

## Download data
# Producer Price Index by Commodity: Chemicals and Allied Products: Industrial Chemicals (WPU061)
chemicals = fred.get_series('WPU061')

# Producer Price Index by Industry: Computer and Electronic Product Manufacturing (PCU334334)
electronics = fred.get_series('PCU334334')

# Producer Price Index by Commodity: Nonmetallic Mineral Products (WPU13)
nonmetallic_mineral = fred.get_series('WPU13')

# Producer Price Index by Commodity: Metals and Metal Products: Iron and Steel (WPU101)
metals = fred.get_series('WPU101')

# Producer Price Index by Commodity: Rubber and Plastic Products: Rubber and Rubber Products (WPU071)
plastics = fred.get_series('WPU071')

# Producer Price Index by Commodity: Pulp, Paper, and Allied Products: Writing and Printing Papers (WPU091301)
pulp_paper = fred.get_series('WPU091301')

# Producer Price Index by Commodity: Textile Products and Apparel: Finished Cotton Broadwoven Fabrics (WPU034201)
textiles = fred.get_series('WPU034201')

## Save data
chemicals.to_csv("../data/chemicals.csv", index = False)
electronics.to_csv("../data/electronics.csv", index = False)
nonmetallic_mineral.to_csv("../data/nonmetallic_mineral.csv", index = False)
metals.to_csv("../data/metals.csv", index = False)
plastics.to_csv("../data/plastics.csv", index = False)
pulp_paper.to_csv("../data/pulp_paper.csv", index = False)
textiles.to_csv("../data/textiles.csv", index = False)

## Get PPI for each prodinit_ippu
ppi_mean = { 'prodinit_ippu_chemicals_tonne' : chemicals.loc['2014-01-01':'2014-12-01'].mean(),
             'prodinit_ippu_electronics_tonne' : electronics.loc['2014-01-01':'2014-12-01'].mean(),
             'prodinit_ippu_lime_and_carbonite_tonne' : nonmetallic_mineral.loc['2014-01-01':'2014-12-01'].mean(),
             'prodinit_ippu_metals_tonne' : metals.loc['2014-01-01':'2014-12-01'].mean(),
             'prodinit_ippu_paper_tonne' : pulp_paper.loc['2014-01-01':'2014-12-01'].mean(),
             'prodinit_ippu_plastic_tonne' : plastics.loc['2014-01-01':'2014-12-01'].mean(),
             'prodinit_ippu_textiles_tonne' :  textiles.loc['2014-01-01':'2014-12-01'].mean()
            }

## Load IO Tables CEPAL_LAC_2014_40s.xlsx
# source : https://www.cepal.org/en/events/global-input-output-tables-tools-analysis-integration-latin-america-world
io = pd.read_excel("../data/CEPAL_LAC_2014_40s.xlsx")

# Load country_iso_3 correspondence
with open(r'../data/countries_correspondence.yaml') as file:
    country_iso_3 = yaml.load(file, Loader=yaml.FullLoader)

country_iso_3 = country_iso_3["country_iso_3"]
country_iso_3_switch = {y: x for x, y in country_iso_3.items()}

# Check LAC countries in CEPAL_LAC_2014_40s
countries_in = set(io["Country_iso3"]).intersection(set(country_iso_3.values()))
countries_out = set(country_iso_3.values()).difference(countries_in)

# Load prodinit_ippu variables equivalence CEPAL_LAC_2014_40s sectors
with open(r'../data/prodinit_ippu_variables.yaml') as file:
    prodinit_ippu_var = yaml.load(file, Loader=yaml.FullLoader)

prodinit_ippu_var = prodinit_ippu_var["prodinit_ippu_var"]

# Compute real prodinit_ippu_var by each country. Original prices: current prices (USD million)
prices_ippu_variables = {'prodinit_ippu_chemicals_tonne': 650,
                         'prodinit_ippu_electronics_tonne': 155,
                         'prodinit_ippu_lime_and_carbonite_tonne': 184,
                         'prodinit_ippu_metals_tonne': 402.55,
                         'prodinit_ippu_paper_tonne': 226.60,
                         'prodinit_ippu_plastic_tonne': 930,
                         'prodinit_ippu_textiles_tonne': 1500}
prodinit_ippu_countries = {c:{var:0 for var in prodinit_ippu_var.keys()} for c in country_iso_3.keys() }

for country,iso in country_iso_3.items():
    if iso in countries_in:
        for var,sector in prodinit_ippu_var.items():
            io_country = io.query("Country_iso3 == '{}'".format(iso))
            consulta = [True if i in prodinit_ippu_var[var] else False for i in  io_country["Nosector"]]
            output_sector = sum(io_country[consulta]["Output"])
            prodinit_ippu_countries[country][var] = (output_sector/ppi_mean[var])*(1000000/prices_ippu_variables[var])  if not math.isnan(output_sector/ppi_mean[var]) else 0

# Impute mean value in contries with no data
prodinit_ippu_var_mean = {k:0 for k in prodinit_ippu_var.keys()}
countries_out.add('SUR')

for var in prodinit_ippu_var.keys():
    cuenta = 0
    for country in country_iso_3.keys():
        if prodinit_ippu_countries[country][var]>0:
            prodinit_ippu_var_mean[var] += prodinit_ippu_countries[country][var]
            cuenta += 1
    prodinit_ippu_var_mean[var] = prodinit_ippu_var_mean[var]/cuenta

    for c_out in countries_out:
        prodinit_ippu_countries[country_iso_3_switch[c_out]][var] = prodinit_ippu_var_mean[var]

# Export csv prodinit_ippu_variables
for var in prodinit_ippu_var.keys():
    df_prodinit_ippu_var = pd.DataFrame()
    for country in country_iso_3.keys():
        partial_df_prodinit_ippu_var = pd.DataFrame.from_dict({'Year':range(1990,2020),'Nation':[country]*30,var:[prodinit_ippu_countries[country][var]]*30})
        df_prodinit_ippu_var = pd.concat([df_prodinit_ippu_var,partial_df_prodinit_ippu_var])
    df_prodinit_ippu_var.to_csv("../output/{}.csv".format(var),index =  False)

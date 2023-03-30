import pandas as pd 

# Load CO2 emission data from climate watch 
ghg_ippu = pd.read_csv("historical_emissions_ippu.csv")

# From short to long
ghg_ippu = pd.melt(ghg_ippu.drop(columns = ['Data source', 'Unit'] , axis = 1), id_vars= ["Country", "Sector", "Gas"])

# Replace NaN by 0
ghg_ippu = ghg_ippu.fillna(0)


# Load ISO3 code
iso3_m49_correspondence = pd.read_html("https://unstats.un.org/unsd/methodology/m49/")[0]

iso3_m49_correspondence.rename(columns = {"Country or Area" : "Country", "ISO-alpha3 code" : "iso_code3"}, inplace = True)

cambia_nombre = {"Bolivia (Plurinational State of)" : "Bolivia", "Venezuela (Bolivarian Republic of)" : "Venezuela"}
iso3_m49_correspondence["Country"] = iso3_m49_correspondence["Country"].replace(cambia_nombre)

iso3_m49_correspondence = iso3_m49_correspondence[["Country", "iso_code3"]]

ghg_ippu = ghg_ippu.merge(right=iso3_m49_correspondence, how = 'inner', on = 'Country')

ghg_ippu["Country"] = ghg_ippu["Country"].apply(lambda x: x.replace(" ","_").lower())
ghg_ippu["Sector"] = ghg_ippu["Sector"].replace({"Industrial Processes" : "IPPU"})
ghg_ippu["value"] = ghg_ippu["value"]*1000

ghg_ippu = ghg_ippu.rename(columns = {"Country" : "Nation", "Sector" : "model", "variable" : "Year"})

ghg_ippu = ghg_ippu[["Nation", "Year", "value",	"model", "iso_code3"]]

ghg_ippu = ghg_ippu.sort_values(["Nation", "Year"])

ghg_ippu.to_csv("ghg_LAC_ippu_co2_iso_code3.csv", index = False)

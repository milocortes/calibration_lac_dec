import pandas as pd
import os

# Change dir
os.chdir("../data")

# Define dataframe to save results
pd_co2_all = pd.DataFrame()

# Load AFOLU CO2 data
afolu_co2 = pd.read_csv("afolu_data_calib_output.csv")
afolu_co2.query("Item=='AFOLU'",inplace = True)
afolu_co2 = afolu_co2[["Year","Area","value"]]
afolu_co2.rename(columns={i:j for i,j in zip(["Year","Area","value"],["year","country","value"])}, inplace=True)
afolu_co2["model"] = "AFOLU"
afolu_co2["country"] = afolu_co2["country"].apply(lambda x: x.lower().replace(" ","_"))

# Load waste CO2 data
waste_co2 = pd.read_csv("waste_ws_datos_imputados.csv")
waste_co2.query("category == 'forecasted'",inplace=True)
waste_co2.drop(columns=["category"],inplace=True)
waste_co2["model"] = "CircularEconomy"

pd_co2_all = pd.concat([pd_co2_all,afolu_co2,waste_co2])

# replace bolivia name
pd_co2_all["country"].replace("bolivia_(plurinational_state_of)","bolivia", inplace=True)

# Change dir
os.chdir("../output")
pd_co2_all.to_csv("co2_all_models.csv", index = False)

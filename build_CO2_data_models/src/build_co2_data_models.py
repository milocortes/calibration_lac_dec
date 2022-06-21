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

# Load ippu CO2 data
ippu_co2 = pd.read_csv("ippu_ws_datos_imputados.csv")
ippu_co2.query("category == 'forecasted'",inplace=True)
ippu_co2.drop(columns=["category"],inplace=True)
ippu_co2["model"] = "IPPU"

# Add data to
# 2003,Suriname,21
# 1990,Trinidad and Tobago,622
# 1999,Venezuela (Bolivarian Republic of),6465
waste_co2 = pd.concat([waste_co2, pd.DataFrame.from_dict({'year':range(1990,2019),'value' : [21] * 29, 'country' : ['suriname'] * 29, 'model' : ['CircularEconomy'] * 29})  ])
waste_co2 = pd.concat([waste_co2, pd.DataFrame.from_dict({'year':range(1990,2019),'value' : [622] * 29 , 'country' : ['trinidad_and_tobago'] * 29, 'model' : ['CircularEconomy'] * 29})  ])
waste_co2 = pd.concat([waste_co2, pd.DataFrame.from_dict({'year':range(1990,2019),'value' : [6465] * 29, 'country' : ['venezuela'] * 29, 'model' : ['CircularEconomy'] * 29})  ])

pd_co2_all = pd.concat([pd_co2_all,afolu_co2,waste_co2,ippu_co2])

# replace bolivia name
pd_co2_all["country"].replace("bolivia_(plurinational_state_of)","bolivia", inplace=True)

# Change dir
os.chdir("../output")
pd_co2_all.to_csv("co2_all_models.csv", index = False)

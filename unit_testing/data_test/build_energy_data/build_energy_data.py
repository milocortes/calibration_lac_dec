import pandas as pd
import os
import glob
from tqdm import tqdm

# Set enery data path
path_energy_data = "/home/milo/Documents/egap/SISEPUEDE/sisepuede_data/Energy"

# Get all variables in directory
variables_energy = [i.split("/")[-2] for i in glob.glob(path_energy_data+"/*/") if i.split("/")[-2] not in ['code', 'raw_data', 'input_to_sisepuede']]

lac_countries = ['argentina', 'bahamas', 'barbados', 'belize', 'bolivia', 'brazil',
       'chile', 'colombia', 'costa_rica', 'dominican_republic', 'ecuador',
       'el_salvador', 'guatemala', 'guyana', 'haiti', 'honduras',
       'jamaica', 'mexico', 'nicaragua', 'panama', 'paraguay', 'peru',
       'suriname', 'trinidad_and_tobago', 'uruguay', 'venezuela']

agrega_df = []
agrega_df_no_rango_total = []
agrega_df_no_todo_lac = []

for var_ener in tqdm(variables_energy):

    if var_ener != "nemomod_entc_reserve_margin":
        print(var_ener)
        historical = pd.read_csv(os.path.join(path_energy_data, var_ener, "input_to_sisepuede", "historical", f"{var_ener}.csv"))
        historical = historical.query("Year >=2011").reset_index(drop = True)
        projected = pd.read_csv(os.path.join(path_energy_data, var_ener, "input_to_sisepuede", "projected", f"{var_ener}.csv"))
        projected = projected.query("Year <= 2050").reset_index(drop = True)

        df_var_ener = pd.concat([historical,projected], ignore_index = True)
        if "Country" in list(df_var_ener.columns):
            df_var_ener.rename(columns = {'Country':'Nation'}, inplace = True)

        df_var_ener["Nation"] = df_var_ener["Nation"].apply(lambda x : x.lower().replace(" ", "_"))
        df_var_ener = df_var_ener[df_var_ener["Nation"].isin(lac_countries)]

        if len(df_var_ener.Nation.unique()) == 26:
            if len(df_var_ener.Year.unique())==40:
                agrega_df.append((var_ener, df_var_ener))
            else:
                agrega_df_no_rango_total.append(df_var_ener)
        else:
            agrega_df_no_todo_lac.append(df_var_ener)

agrupa_df = []

for var_energ,df_agrega in agrega_df:
    df_agrega =  df_agrega.rename(columns = {var_energ : "value"})
    df_agrega["variable"] = var_energ
    agrupa_df.append(df_agrega)

all_data_energy = pd.concat(agrupa_df, ignore_index = True)
all_data_energy = all_data_energy[["Year", "Nation", "value", "variable"]]

all_data_energy_sorted = all_data_energy.sort_values(by= ["Year", "Nation"])

all_data_energy_pivot = all_data_energy_sorted.pivot(index = ['Year', 'Nation'], columns = 'variable', values = 'value').reset_index()
all_data_energy_pivot = all_data_energy_pivot.sort_values(by=["Nation","Year"]).reset_index(drop=True)
all_data_energy_pivot.to_csv("energy_data_lac.csv", index = False)


##########################################

import pandas as pd
import os

energy_data = pd.read_csv("energy_data_lac.csv")

data_path = "/home/milo/Documents/egap/SISEPUEDE/packaging_projects/github_projects/sisepuede_calibration_dev/unit_testing/data_test"

df_input_all_countries = pd.read_csv( os.path.join(data_path, "real_data_2023_01_11.csv"))

df_input_all_countries.set_index(["Year","Nation"], inplace = True)
energy_data.set_index(["Year","Nation"],inplace = True)

quita_energia = list(set(df_input_all_countries.columns).intersection(energy_data.columns))

df_input_all_countries = df_input_all_countries.drop(columns=quita_energia)
df_input_all_countries = pd.concat([df_input_all_countries,energy_data], axis = 1)
df_input_all_countries = df_input_all_countries.reset_index()

df_input_all_countries.to_csv("real_data_2023_01_11_energy.csv", index = False)

import sys
import pandas as pd

sector = sys.argv[1]
country = sys.argv[2]

sector_equiv = {"AFOLU" : "afolu", "IPPU" : "ippu", "CircularEconomy" : "circular_economy"}
sector_file = sector_equiv[sector]

fake_data_sector = pd.read_csv("lac_decarbonization/ref/fake_data/fake_data_{}.csv".format(sector_file))

vars_sector = fake_data_sector.columns[2:]

observed_vars_sector = pd.read_csv("observed_data/resumen_variables_observadas.csv")
observed_vars_sector = observed_vars_sector.query("sector=='{}'".format(sector))["variable"].to_list()

df_observed_vars = pd.read_csv("all_countries_test_CalibrationModel_class.csv")
df_observed_vars.query("country =='{}'".format(country), inplace = True)

for i in [x for x in observed_vars_sector if x in vars_sector]:
    print("Variable\n{}\n".format(i))
    print("\nMean value fake data ------ Mean value observed data/n")
    mean_fake_data = fake_data_sector[i].mean()
    mean_observed_data = df_observed_vars[i].mean()
    print("         {}                               {}      \n".format(mean_fake_data,mean_observed_data))
    try:
        print("Ratio between observed/fake     {}\n".format(mean_observed_data/mean_fake_data))
    except:
        print("0 in fake variable")
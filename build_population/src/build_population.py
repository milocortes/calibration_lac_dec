import pandas as pd

countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador','el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']

year_init,year_end = 1990,2019

def get_data_frame(csv_name):
    df = pd.read_csv("../data/{}".format(csv_name))
    df.drop(columns=['Country Code','Indicator Name', 'Indicator Code'], inplace=True)
    df.rename(columns = {'Country Name':'Nation'}, inplace = True)
    df = pd.melt(df, id_vars='Nation', value_vars=[str(i) for i in range(1960,2022)])

    df.rename(columns = {'variable' : 'Year'} , inplace = True)

    df["Nation"] = df["Nation"].apply(lambda x :x.lower().replace(" ","_"))
    df["Nation"] = df["Nation"].apply(lambda x :x.replace(",_the",""))


    df = df[df["Nation"].apply(lambda x :x in countries_list)]

    df["Year"] = pd.to_numeric(df["Year"])
    return df.query("(Year >= {} and Year <= {} )".format(year_init,year_end))

population_rural = get_data_frame("rural_population_share.csv")
population_urban = get_data_frame("urban_population_share.csv")
population = get_data_frame("population.csv")

population_rural["value"] = (population_rural["value"]/100) * population["value"]
population_urban["value"] = (population_urban["value"]/100) * population["value"]

population_rural["value"] = population_rural["value"].apply(lambda x : int(x))
population_urban["value"] = population_urban["value"].apply(lambda x : int(x))

population_rural.rename(columns = {'value':'population_rural'}, inplace = True)
population_urban.rename(columns = {'value':'population_urban'}, inplace = True)



population_rural.to_csv("../../observed_data/population_rural.csv",index = False)
population_urban.to_csv("../../observed_data/population_urban.csv",index = False)

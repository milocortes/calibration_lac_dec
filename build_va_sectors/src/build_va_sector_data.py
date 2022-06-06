import pandas as pd

countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador','el_salvador','guatemala','guyana','haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']


year_init,year_end = 2006,2014

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

gdp_data = get_data_frame('gdp_1960_2020_constant.csv')
gdp_data["value"] = gdp_data["value"]/(1000**3)
manu_data = get_data_frame('manu_share_gdp.csv')
mineria_data = get_data_frame('mineria_share_gdp.csv')
service_data = get_data_frame('service_share_gdp.csv')
industry_data = get_data_frame('industry_share_gdp.csv')

manu_data.loc[manu_data["Nation"]=="barbados", 'value'] = manu_data.query("Nation !='barbados'")[["Year","value"]].groupby("Year").mean().value.to_numpy()
service_data.loc[service_data["Nation"]=="barbados", 'value'] = service_data.query("Nation !='barbados'")[["Year","value"]].groupby("Year").mean().value.to_numpy()
industry_data.loc[industry_data["Nation"]=="barbados", 'value'] = industry_data.query("Nation !='barbados'")[["Year","value"]].groupby("Year").mean().value.to_numpy()

manu_data["value"] = (manu_data["value"]/100) * gdp_data["value"]
industry_data["value"] = ((industry_data["value"]/100) - (manu_data["value"]/100) ) * gdp_data["value"]
service_data["value"] = (service_data["value"]/100) * gdp_data["value"]
mineria_data["value"] = (mineria_data["value"]/100) * gdp_data["value"]
comercio_data = gdp_data.copy()
comercio_data["value"] = (1 - ((industry_data["value"]/100) - (manu_data["value"]/100) ) - (manu_data["value"]/100) - (service_data["value"]/100) - (mineria_data["value"]/100)) * gdp_data["value"]


manu_data.rename(columns = {'value':'va_manufacturing_mmm_usd'} , inplace = True)
industry_data.rename(columns = {'value':'va_industrial_mmm_usd'}, inplace = True)
#service_data.rename({'value':})
mineria_data.rename(columns = {'value':'va_mining_mmm_usd'}, inplace = True)
comercio_data.rename(columns = {'value':'va_commercial_mmm_usd'}, inplace = True)

manu_data.to_csv("../../observed_data/va_manufacturing_mmm_usd.csv",index = False)
industry_data.to_csv("../../observed_data/va_industrial_mmm_usd.csv",index = False)
comercio_data.to_csv("../../observed_data/va_commercial_mmm_usd.csv",index = False)
mineria_data.to_csv("../../observed_data/va_mining_mmm_usd.csv",index = False)



#iso_code = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_latest.csv")

#gdp_data.rename(columns = {"iso_code3":"iso_code"}, inplace = True)

#gdp_data = pd.merge(gdp_data,iso_code[['iso_code','entity']].rename(columns = {"entity" : "Nation"}),on = 'iso_code', how = 'left')


#https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv

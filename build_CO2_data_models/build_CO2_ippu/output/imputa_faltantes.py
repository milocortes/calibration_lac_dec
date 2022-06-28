import pandas as pd
from scipy import stats
import math


co2_ippu = pd.read_csv("ippu_ws_datos_imputados.csv")
gdp = pd.read_csv("../../../observed_data/gdp_mmm_usd.csv")
pop = pd.read_csv("../../../observed_data/population_gnrl_urban.csv")

co2_ippu_belize = co2_ippu.query("country =='belize' and (year >= 2006 and year <=2014) and category =='imputed'")["value"].values
#co2_ippu_belize = co2_ippu.query("country =='belize' and (year >= 2006 and year <=2014) and category =='imputed'")["value"].apply(lambda x: math.log(x)).values
gdp_belize = gdp.query("Nation =='belize'")
gdp_belize["gdp_per_capita"] = gdp.query("Nation =='belize'")["gdp_mmm_usd"].values*1000000000 / pop.query("Nation =='belize'")["population_gnrl_urban"].values
#gdp_belize["gdp_per_capita"] = gdp_belize["gdp_per_capita"].apply(lambda x: math.log(x))

slope, intercept, r, p, std_err = stats.linregress(gdp_belize["gdp_per_capita"].values, co2_ippu_belize)

paises_imputa = ['bahamas','guyana','haiti']

acumula_co2_imputados = pd.DataFrame()

for pais in paises_imputa:
    gdp_pais = gdp.query("Nation =='{}'".format(pais))
    pop_pais = pop.query("Nation =='{}'".format(pais))
    gdp_per_capita = gdp_pais["gdp_mmm_usd"].values*1000000000 / pop_pais["population_gnrl_urban"].values
    df_gdp_per_capita_imputed = pd.DataFrame.from_dict({"year":range(2006,2015),'country':[pais]*9,'value':gdp_per_capita*slope,'category':['imputed']*9})
    df_gdp_per_capita_observed = pd.DataFrame.from_dict({"year":range(2006,2015),'country':[pais]*9,'value':gdp_per_capita*slope,'category':['observed']*9})
    df_gdp_per_capita_forecasted = pd.DataFrame.from_dict({"year":range(2006,2015),'country':[pais]*9,'value':gdp_per_capita*slope,'category':['forecasted']*9})
    acumula_co2_imputados = pd.concat([acumula_co2_imputados,df_gdp_per_capita_imputed,df_gdp_per_capita_observed,df_gdp_per_capita_forecasted])

co2_ippu = pd.concat([co2_ippu,acumula_co2_imputados])
co2_ippu.to_csv("ippu_ws_datos_imputados.csv",index=False)

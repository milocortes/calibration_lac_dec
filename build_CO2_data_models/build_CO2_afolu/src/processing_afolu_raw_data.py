import pandas as pd

afolu = pd.read_csv("../data/afolu_data_calib_output.csv")

afolu["Area"] = afolu["Area"].apply(lambda x : x.lower().replace(" ","_"))

#countries_list = ['argentina','bahamas','barbados','belize','bolivia','brazil','chile','colombia','costa_rica','dominican_republic','ecuador','el_salvador','guatemala','guyana',
#                  'haiti','honduras','jamaica','mexico','nicaragua','panama','paraguay','peru','suriname','trinidad_and_tobago','uruguay','venezuela']

df_agrupa = pd.DataFrame()

for c in set(afolu.Area):
    co2_country = afolu.query("Item=='AFOLU' and (Year >=2011 and Year <=2019) and Area =='{}'".format(c))["value"].values
    if c =='chile' or c =='costa_rica':
        co2_country = co2_country*-1
    n = len(range(2011,2020))
    df_parcial = pd.DataFrame.from_dict({'year':range(2011,2020),
                                         'country': [c]*n,
                                         'value': co2_country})
    df_agrupa = pd.concat([df_agrupa,df_parcial])

df_agrupa.to_csv("../output/afolu_co2_lac_data.csv", index = False)

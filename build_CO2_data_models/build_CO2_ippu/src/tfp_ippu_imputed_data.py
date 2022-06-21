import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

# Prevenimos que tensorflow asigne la totalidad de la memoria de la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

df = pd.read_csv("../data/ippu_ts_data.csv")
df.fillna(np.nan,inplace=True)
df["year"] =  range(1990,2020)

imputed_mean = []

for c in df.columns:
    if df[c].isna().sum() >= 28:
        imputed_mean.append(c)
        promedio = df[c].mean(skipna=True)
        df[c] = df[c].apply(lambda x :promedio if np.isnan(x) else x)

df_imputed_mean = pd.melt(df[["year"]+imputed_mean],id_vars = 'year', value_vars=imputed_mean)
df_imputed_mean.rename(columns={'variable':'country'},inplace=True)

df_imputed_total = pd.DataFrame()

for cat in ['imputed','forecasted','observed']:
        df_imputed_mean["category"] = cat
        df_imputed_total = pd.concat([df_imputed_total,df_imputed_mean])

for country in set(df.columns).difference(imputed_mean):
    print(country)
    try:
        time_series_with_nans = df[country].to_numpy()
        observed_time_series = tfp.sts.MaskedTimeSeries(
          time_series=time_series_with_nans,
          is_missing=tf.math.is_nan(time_series_with_nans))

        # Build model using observed time series to set heuristic priors.
        linear_trend_model = tfp.sts.LocalLinearTrend(
          observed_time_series=observed_time_series)

        model = tfp.sts.Sum([linear_trend_model],
                            observed_time_series=observed_time_series)

        # Fit model to data
        parameter_samples, _ = tfp.sts.fit_with_hmc(model, observed_time_series)

        # Forecast
        forecast_dist = tfp.sts.forecast(
          model, observed_time_series,parameter_samples=parameter_samples, num_steps_forecast=5)

        # Impute missing values
        observations_dist = tfp.sts.impute_missing_values(model, observed_time_series,parameter_samples=parameter_samples)

        # add imputed data to real data
        time_series_with_nans_imputed = time_series_with_nans
        pronosticado = list(np.array(observations_dist.mean()))

        for k,v in enumerate(time_series_with_nans_imputed):
            if np.isnan(v):
                time_series_with_nans_imputed[k] = pronosticado[k]

        dict_time_series_with_nans_imputed = pd.DataFrame.from_dict({"year": [i for i in range(1990,2020)],"value":time_series_with_nans_imputed,"country":[country for i in range(1990,2020)],"category":["imputed" for i in range(1990,2020)]})
        dict_time_series_with_nans_forecasted = pd.DataFrame.from_dict({"year": [i for i in range(1990,2020)],"value":pronosticado,"country":[country for i in range(1990,2020)],"category":["forecasted" for i in range(1990,2020)]})
        dict_time_series_with_nans_observed = pd.DataFrame.from_dict({"year": [i for i in range(1990,2020)],"value":time_series_with_nans,"country":[country for i in range(1990,2020)],"category":["observed" for i in range(1990,2020)]})

        df_imputed_total = pd.concat([df_imputed_total,dict_time_series_with_nans_imputed,dict_time_series_with_nans_forecasted,dict_time_series_with_nans_observed])
    except Exception as e:
        print("Execution fail. Country: {}".format(country))


df_imputed_total.to_csv("../output/ippu_ws_datos_imputados.csv",index=False)

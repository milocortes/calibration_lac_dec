import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

# Prevenimos que tensorflow asigne la totalidad de la memoria de la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

df = pd.read_csv("/home/milo/Documents/egtp/LAC-dec/calibration/build_CO2_data_models/build_CO2_waste/data/waste_ts_data.csv")
df.fillna(np.nan,inplace=True)

df_imputed_total = pd.DataFrame()

for country in df.columns:
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

        dict_time_series_with_nans_imputed = pd.DataFrame.from_dict({"year": [i for i in range(1990,2019)],"value":time_series_with_nans_imputed,"country":[country for i in range(1990,2019)],"category":["imputed" for i in range(1990,2019)]})
        dict_time_series_with_nans_forecasted = pd.DataFrame.from_dict({"year": [i for i in range(1990,2019)],"value":pronosticado,"country":[country for i in range(1990,2019)],"category":["forecasted" for i in range(1990,2019)]})
        dict_time_series_with_nans_observed = pd.DataFrame.from_dict({"year": [i for i in range(1990,2019)],"value":time_series_with_nans,"country":[country for i in range(1990,2019)],"category":["observed" for i in range(1990,2019)]})

        df_imputed_total = pd.concat([df_imputed_total,dict_time_series_with_nans_imputed,dict_time_series_with_nans_forecasted,dict_time_series_with_nans_observed])
    except Exception as e:
        print("Execution fail. Country: {}".format(country))


df_imputed_total.to_csv("/home/milo/Documents/egtp/LAC-dec/calibration/build_CO2_data_models/build_CO2_waste/output/waste_ws_datos_imputados.csv",index=False)

"""
import matplotlib.pyplot as plt

xs = np.arange(len(time_series_with_nans))
series1 = np.array(time_series_with_nans).astype(np.double)
s1mask = np.isfinite(series1)
imputacion = list(np.array(observations_dist.mean()))
plt.plot(xs, imputacion, linestyle='-', marker='o',color="blue")
plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o',color="red")
plt.show()
"""

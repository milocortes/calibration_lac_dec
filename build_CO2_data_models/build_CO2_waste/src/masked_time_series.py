import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

time_series_with_nans = [-1., 1., np.nan, 2.4, np.nan, np.nan]
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
print('imputed means and stddevs: ',
      observations_dist.mean(),
      observations_dist.stddev())


import matplotlib.pyplot as plt

xs = np.arange(6)
series1 = np.array(time_series_with_nans).astype(np.double)
s1mask = np.isfinite(series1)
imputacion = list(np.array(observations_dist.mean()))
plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o',color="blue")
plt.plot(xs, imputacion, linestyle='-', marker='o',color="red")
plt.show()

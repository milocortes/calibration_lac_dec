# Calibration-Lac-Decarbonization 

Este repositorio contiene los programas para procesar los datos de insumo del modelo LAC-decarbonization así como también para realizar la calibración de los sectores que lo componen.

La estructura de directorios es la siguiente:

```
.
├── build_bounds
├── build_CO2_data_models
├── build_ippu
├── build_population
├── build_va_sectors
├── docs
├── observed_data
└── output_calib
```

A continuación se describe cada uno de estos:
* ```build_CO2_data_models``` contiene los programas y datos de insumo para construir los datos de emisiones de CO2 por sector. 

* ```build_ippu``` contiene los programas y datos de insumo para construir los datos de producción inicial del sector IPPU.
* ```build_population``` contiene los programas y datos de insumo para construir los datos de ```population_gnrl_rural``` y ```population_gnrl_urban```.
* ```build_va_sectors``` contiene los programas y datos de insumo para construir los datos de valor agregado de los sectores de manufactura, industria, comercio, minería así como el gdp.
* ```observed_data``` contiene los datos observados de los sectores del modelo.
* ```output_calib``` recibe los datos de salida de la calibración del modelo por sector. 
* ```build_bounds``` construye las bandas inferiores y superiores de los vectores de calibración de los sectores.
 
 
A continuación se describirán los pasos para construir los datos de insumo para la calibración de modelo LAC-decarbonization.

## Construcción de resumen de variables observadas

Dentro del directorio ```observed_data``` se encuetra el programa ```build_resumen.py``` el cual construye una tabla (```resumen_variables_observadas.csv```) en la que se lista las variables observadas por sector hasta el momento, así como información adicional como si dicha variable efectivamente es un insumo del modelo, el periodo de inicio y final de la variable, entre otras. 

Desde el directorio raíz podemos ejecutar la siguiente instrucción para generar ```resumen_variables_observadas.csv```:
```
python observed_data/build_resumen.py 
```

## Agrega datos observados a los datos fake

El archivo ```add_observed_data.py``` agrega los datos observados de cada sector al conjunto de datos fake que funcionan como insumos al modelo. 


Desde el directorio raíz podemos ejecutar la siguiente instrucción para generar el archivo ```all_countries_test_CalibrationModel_class.csv``` el cual integra los datos observados:
```
python add_observed_data.py 
```

## Verifica consistencia entre datos observados y datos fake

El programa ```verify_data_consistency.py``` imprime los valores medios de las variables observadas del sector y de los valores medios de las variables fake. Muestra la razón entre el valor medio observado y el valor medio fake. El programa recibe dos argumentos suministrados desde la ejecución en linea de comandos: el sector y el país. 

Un ejemplo de la ejecución del programa es el siguiente:
```
python verify_data_consistency.py AFOLU argentina 
```
## Calibración de prueba

Los archivos ```test_AFOLU_class.py``` y ```test_IPPU_class.py``` ejecutan una prueba de calibración tomando como referencia un país y un sector. La clase ```CalibrationModel``` es la encargada de realizar la calibración. Dicha clase a su vez extiende de la clase ```RunModel```, la cual se inicializa con los siguientes argumentos:

```
------------------------------------------

RunModel class

-------------------------------------------

# Inputs:
    * df_input_var              - Pandas DataFrame of input variables
    * country                   - Country of calibration
    * calib_targets             - Variables that will be calibrated
    * t_times                   - List with the period of simulation
    * subsector_model           - Subsector model
    * downstream                - Flag that says if the model is running integrated (i.e. with the output of latest subsector model )
    * best_vector

```

Por su parte, la clase ```CalibrationModel``` se inicializa con los siguientes argumentos:

```
------------------------------------------

CalibrationModel class

-------------------------------------------

# Inputs:
    * df_calib_bounds           - Pandas DataFrame with the calibration scalar ranges
    * t_times                   - List with the period of simulation
    * df_co2_emissions          - Pandas DataFrame of co2_emissions
    * cv_calibration            - Flag that says if the calibration runs cross validation
    * cv_training               - List with the periods of the training set
    * cv_test                   - List with the periods of the test set


# Output
    * mse_all_period            - Mean Squared Error for all periods
    * mse_training              - Mean Squared Error for training period
    * mse_test                  - Mean Squared Error for test period
    * calib_vector              - List of calibration scalar
```


El siguiente fragmento de código se encuentra en el programa ```test_AFOLU_class.py``` y muestra un ejemplo de la calibración para un país para el sector AFOLU:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration_lac import RunModel,CalibrationModel

df_input_all_countries = pd.read_csv("all_countries_test_CalibrationModel_class.csv")

# Define target country
target_country = "colombia"

# Set model to run
models_run = "AFOLU"

# Load observed CO2 data
df_co2_observed_data = pd.read_csv("build_CO2_data_models/output/co2_all_models.csv")

# Load calib targets by model to run
df_calib_targets =  pd.read_csv("build_bounds/output/calib_bounds_sector.csv")

calib_bounds = df_calib_targets.query("sector =='{}'".format(models_run))
calib_targets = calib_bounds['variable']

# Define lower and upper time bounds
year_init,year_end = 2011,2019

df_input_country = df_input_all_countries.query("country =='{}'".format(target_country)).reset_index().drop(columns=["index"])
t_times = range(year_init, year_end+1)
df_co2_observed_data = df_co2_observed_data.query("model == '{}' and country=='{}' and (year >= {} and year <= {} )".format(models_run,target_country,year_init,year_end))

calibration = CalibrationModel(df_input_country, target_country, models_run,
                                calib_targets, calib_bounds, t_times,
                                df_co2_observed_data,cv_training = [0,1,2,3,4,5,6,7,8] , cv_test = [1,3,5,7], cv_calibration = True)

X = [np.mean((calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "min_35"].item(),calibration.df_calib_bounds.loc[calibration.df_calib_bounds["variable"] == i, "max_35"].item()))  for i in calibration.calib_targets["AFOLU"]]

calibration.f(X)
calibration.run_calibration("genetic_binary", population = 60, maxiter = 20)
plt.plot(calibration.fitness_values["AFOLU"])
plt.show()


output_data = calibration.get_output_data(calibration.best_vector["AFOLU"])
co2_computed = output_data[calibration.var_co2_emissions_by_sector["AFOLU"]].sum(axis=1)
plt.plot([i/1000 for i in df_co2_observed_data.value],label="historico")
plt.plot(co2_computed,label="estimado")
plt.title(target_country)
plt.legend()
plt.show()
```

import random
import math
import numpy as np
from multiprocessing import Pool,cpu_count
import time
import pandas as pd

# Load LAC-Decarbonization source
import sys
import os
cwd = os.getcwd()

sys.path.append(cwd + '/lac_decarbonization/python')

import data_structures as ds
import setup_analysis as sa
import support_functions as sf
import sector_models as sm
import argparse

from model_socioeconomic import Socioeconomic
from model_afolu import AFOLU
from model_circular_economy import CircularEconomy
from model_ippu import IPPU

# Genera población aleatoria binaria de m bit-string y cromosomas de tamaño n
def rand_population_binary(m,n):
    return [[random.randint(0, 1) for j in range(n)]for i in range(m)]

# Función que codifica las variables
def length_variable(i_sup,i_inf,precision):
    return int(math.ceil(math.log2((i_sup-i_inf)*10**(precision))))

# Función que obtiene las potencias en base dos de un vector de bits
def to_decimal(dimension,v):
    v.reverse()
    return sum(np.array([2**(i) for i in range(dimension)])*np.array(v))

# Función que codifica el vector de bits a un valor real
def binary2real(i_sup,i_inf,dimension,pob):
     return [i_inf + (to_decimal(dimension,v)*(i_sup-i_inf)/(2**(dimension)-1)) for v in pob]

# Función que genera la estructura de datos Fenotipo
def DECODE(n_variables,m,i_sup_vec,i_inf_vec,dimension_vec,pob_vec):

    feno = [[] for i in range(m)]

    for i in range(n_variables):
        i_sup = i_sup_vec[i]
        i_inf = i_inf_vec[i]
        pob = pob_vec[i]
        dim = dimension_vec[i]
        b2r = binary2real(i_sup,i_inf,dim,pob)
        for k in range(m):
            feno[k].append(b2r[k])

    return feno

# Funcion que genera la estructura de datos de la función objetivo
def OBJFUN(f,feno,bandera,procesos):
    if bandera == True:
        nproc = cpu_count()
        p = Pool(nproc-2)
        with p:
            resultado = p.map(f, feno)
        #return list(map(ackley,feno))
        return resultado
    else:
        p = Pool(procesos)
        with p:
            resultado = p.map(f, feno)
        #return list(map(ackley,feno))
        return resultado
# Función que genera la aptitud de los individuos
def APTITUD(objv,operacion):

    val_max = max(objv)
    val_min = min(objv)

    if operacion == "min":
        objv_norm = [(((i-val_min)/(val_max-val_min))+0.01)**-1 for i in objv]
        suma = sum(objv_norm)
        key_objv = [(k,i/suma) for (k,i) in enumerate(objv_norm)]
        objv_sort = sorted(key_objv,key=lambda tup: tup[1],reverse=True)

    elif operacion == "max":
        objv_norm = [(((i-val_min)/(val_max-val_min))+0.1) for i in objv]
        suma = sum(objv_norm)
        key_objv = [(k,i/suma) for (k,i) in enumerate(objv_norm)]
        objv_sort = sorted(key_objv,key=lambda tup: tup[1],reverse=True)

    return objv_sort

# Función que selecciona a los mejores individuos
def SELECCION(aptitud,tipo,n_variables,población):
    if tipo == "ruleta":
        n = int(len(aptitud)/2)
        suma_acumulada = np.cumsum([v for (k,v) in aptitud])

        individuos_dict = {i:{} for i in range(n)}

        for pareja in range(n):
            for individuo in range(2):
                aleatorio = random.random()
                index_ind = np.where(suma_acumulada >= aleatorio)[0][0]
                cromosoma = []
                for gen in range(n_variables):
                    cromosoma.append(población[gen][aptitud[index_ind][0]])

                cromosoma = sum(cromosoma,[])
                individuos_dict[pareja][individuo] = cromosoma

    return individuos_dict

def CRUZA(seleccion,tipo,length_total_cromosoma):
    if tipo == "unpunto":
        n = len(seleccion)

        nueva_poblacion = []

        for pareja in range(n):
            punto_cruza = random.randint(0, length_total_cromosoma)

            primer_nuevo_individuo = seleccion[pareja][0][0:punto_cruza] + seleccion[pareja][1][punto_cruza:length_total_cromosoma]
            segundo_nuevo_individuo = seleccion[pareja][1][0:punto_cruza] + seleccion[pareja][0][punto_cruza:length_total_cromosoma]

            nueva_poblacion.append(primer_nuevo_individuo)
            nueva_poblacion.append(segundo_nuevo_individuo)

    return nueva_poblacion

def MUTACION(nueva_poblacion,length_total_cromosoma,n_variables,dimension_vec):

    mutacion_param = 2/length_total_cromosoma
    n = len(nueva_poblacion)

    for individuo in range(n):
         muta_random = np.array([random.random() for i in range(length_total_cromosoma)])
         muta_index = np.where(muta_random < mutacion_param)[0]

         for i in muta_index:
             nueva_poblacion[individuo][i] = int(not nueva_poblacion[individuo][i])

    inicio = 0
    fin = 0
    nueva_poblacion_format = []

    for gen in range(n_variables):
        nueva_poblacion_gen = []
        fin += dimension_vec[gen]
        for individuo in nueva_poblacion:
            nueva_poblacion_gen.append(individuo[inicio:fin])

        nueva_poblacion_format.append(nueva_poblacion_gen)
        inicio +=dimension_vec[gen]

    return nueva_poblacion_format


'''
------------------------------------------

CalibrationModel class

-------------------------------------------

# Inputs:
    * df_input_var              - Pandas DataFrame of input variables
    * country                   - Country of calibration
    * calib_targets             - Variables that will be calibrated
    * df_calib_bounds           - Pandas DataFrame with the calibration scalar ranges
    * t_times                   - List with the period of simulation
    * df_co2_emissions          - Pandas DataFrame of co2_emissions
    * subsector_model           - Subsector model
    * optimization_method       - Optimization method. Args: genetic_binary, differential_evolution, pso
    * cv_calibration            - Flag that says if the calibration runs cross validation
    * cv_training               - List with the periods of the training set
    * cv_test                   - List with the periods of the test set

# Output
    * mse_all_period            - Mean Squared Error for all periods
    * mse_training              - Mean Squared Error for training period
    * mse_test                  - Mean Squared Error for test period
    * calib_vector              - List of calibration scalar
'''

class CalibrationModel(object):
    """docstring for CalibrationModel."""

    def __init__(self, df_input_var, country, subsector_model, calib_targets, df_calib_bounds, t_times,
                 df_co2_emissions, cv_calibration = False, cv_training = [], cv_test = [], cv_run = 0, id_mpi = 0):
        self.df_input_var = df_input_var
        self.country = country
        self.calib_targets = calib_targets
        self.df_calib_bounds = df_calib_bounds
        self.t_times = t_times
        self.df_co2_emissions = df_co2_emissions
        self.subsector_model = subsector_model
        self.cv_calibration = cv_calibration
        self.cv_training = cv_training
        self.cv_test = cv_test
        self.var_co2_emissions_by_sector = {'CircularEconomy' : ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"]}
        self.rest_parameters = list(set(df_input_var) -set(calib_targets))
        self.cv_run = cv_run
        self.id_mpi = id_mpi
        self.fitness_values = {'CircularEconomy':[]}
        self.best_vector = {'CircularEconomy':[]}


    """
    ---------------------------------
    objective_a method
    ---------------------------------

    Description: The method recive params, run the subsector model,
                 and compute the MSE by the specific period of simulation

    # Inputs:
             * params       - calibration vector

    # Output:
             * output       - MSE value

    """
    def objective_a(self, params):

        x_mean = self.df_input_var[self.calib_targets].iloc[self.cv_training].mean() * params
        b1 = pd.DataFrame.from_dict({j:[i]*len(self.cv_training) for i,j in zip(x_mean,self.calib_targets)})
        partial_rest_parameters_df = self.df_input_var[self.rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)

        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(input_pivot)

        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(input_pivot)

        if self.subsector_model == "IPPU":
            #print("\n\tRunning IPPU")
            model_ippu = IPPU(sa.model_attributes)
            df_model_data_project = model_ippu.project(input_pivot)

        #out_vars = df_model_data_project.columns[ ["emission_co2e" in i for i in  df_model_data_project.columns]]
        out_vars = self.var_co2_emissions_by_sector[self.subsector_model]
        model_data_co2e = df_model_data_project[out_vars].sum(axis=1)
        #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
        trend = self.df_co2_emissions.value
        trend = [i/1000 for i in trend]

        if self.cv_calibration:
            model_data_co2e = np.array([model_data_co2e[i] for i in self.cv_training])
            trend = np.array([trend[i] for i in self.cv_training])

        output = np.mean((model_data_co2e-trend)**2)

        return output

    """
    --------------------------------------------
    f method
    --------------------------------------------

    Description : fitness function

    # Inputs:
             * X        - calibration vector

    # Output:
             * output   - MSE value
    """
    def f(self, X):
        return self.objective_a(X)


    """
    ---------------------------------
    get_mse_test method
    ---------------------------------

    Description: The method recive params, run the subsector model,
                 and compute the MSE by the specific period of simulation

    # Inputs:
             * params       - calibration vector

    # Output:
             * output       - MSE value
    """

    def get_mse_test(self, params):
        x_mean = self.df_input_var[self.calib_targets].iloc[self.cv_test].mean() * params
        b1 = pd.DataFrame.from_dict({j:[i]*len(self.cv_test) for i,j in zip(x_mean,self.calib_targets)})
        partial_rest_parameters_df = self.df_input_var[self.rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)

        if self.subsector_model == "AFOLU":
            print("\n\tAFOLU")
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(input_pivot)

        if self.subsector_model == "CircularEconomy":
            print("\n\tRunning CircularEconomy")
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(input_pivot)

        if self.subsector_model == "IPPU":
            print("\n\tRunning IPPU")
            model_ippu = IPPU(sa.model_attributes)
            df_model_data_project = model_ippu.project(input_pivot)

        #out_vars = df_model_data_project.columns[ ["emission_co2e" in i for i in  df_model_data_project.columns]]
        out_vars = self.var_co2_emissions_by_sector[self.subsector_model]
        model_data_co2e = df_model_data_project[out_vars].sum(axis=1)
        #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
        trend = self.df_co2_emissions.value
        trend = [i/1000 for i in trend]

        if self.cv_calibration:
            model_data_co2e = np.array([model_data_co2e[i] for i in self.cv_test])
            trend = np.array([trend[i] for i in self.cv_test])

        output = np.mean((model_data_co2e-trend)**2)

        return output

    """
    ---------------------------------
    get_output_data method
    ---------------------------------

    Description: The method recive params, run the subsector model,
                 and compute output data of specific subsector model

    # Inputs:
             * params                   - calibration vector

    # Output:
             * df_model_data_project    - output data of specific subsector model
    """

    def get_output_data(self, params):
        x_mean = self.df_input_var[self.calib_targets].iloc[self.cv_test].mean() * params
        b1 = pd.DataFrame.from_dict({j:[i]*len(self.cv_test) for i,j in zip(x_mean,self.calib_targets)})
        partial_rest_parameters_df = self.df_input_var[self.rest_parameters]
        input_pivot =  pd.concat([b1.reset_index(drop=True), partial_rest_parameters_df], axis=1)

        if self.subsector_model == "AFOLU":
            print("\n\tAFOLU")
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(input_pivot)

        if self.subsector_model == "CircularEconomy":
            print("\n\tRunning CircularEconomy")
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(input_pivot)

        if self.subsector_model == "IPPU":
            print("\n\tRunning IPPU")
            model_ippu = IPPU(sa.model_attributes)
            df_model_data_project = model_ippu.project(input_pivot)

        return df_model_data_project

    '''
    ------------------------------------------

    run_calibration method

    -------------------------------------------

    # Inputs:
        * optimization_method       - Optimization method. Args: genetic_binary, differential_evolution, pso


    # Output
        * mse_all_period            - Mean Squared Error for all periods
        * mse_training              - Mean Squared Error for training period
        * mse_test                  - Mean Squared Error for test period
        * calib_vector              - List of calibration scalar
    '''

    def run_calibration(self,optimization_method,population, maxiter):
        if optimization_method == "genetic_binary":
            # Optimize the function

            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            n_variables = len(self.calib_targets)
            i_sup_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item() +0.01 for i in self.calib_targets]
            i_inf_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets]
            precision = 8
            m = population
            maxiter = maxiter
            dimension_vec = []
            genotipo = []
            length_total_cromosoma = 0

            ## Generamos población inicial
            for i in range(n_variables):
                length_cromosoma = length_variable(i_sup_vec[i],i_inf_vec[i],precision)
                length_total_cromosoma += length_cromosoma
                dimension_vec.append(length_cromosoma)
                genotipo.append(rand_population_binary(m, length_cromosoma))

            ## Iniciamos el algoritmo genético
            feno = DECODE(n_variables,m,i_sup_vec,i_inf_vec,dimension_vec,genotipo)
            print("Evaluando poblacion inicial")
            objv = OBJFUN(self.f,feno,False,1)

            resultados = []
            mejor_individuo = 0
            mejor_valor = 100000000

            for it in range(maxiter):
                print("-----------------------------")
                print(it)
                print("-----------------------------")

                aptitud = APTITUD(objv,"min")
                seleccion = SELECCION(aptitud,"ruleta",n_variables,genotipo)
                genotipo = CRUZA(seleccion,"unpunto",length_total_cromosoma)
                genotipo = MUTACION(genotipo,length_total_cromosoma,n_variables,dimension_vec)
                feno = DECODE(n_variables,m,i_sup_vec,i_inf_vec,dimension_vec,genotipo)
                objv = OBJFUN(self.f,feno,False,1)
                resultados.append(min(objv))
                mejor_individuo = objv.index(min(objv))
                #print("Mejor valor fun.obj ---> {}. Variables de decision ---> {}".format(objv[mejor_individuo],feno[mejor_individuo]))
                #print("Mejor valor fun.obj ---> {}".format(objv[mejor_individuo]))

                if objv[mejor_individuo] < mejor_valor:
                    mejor_valor = objv[mejor_individuo]
                    mejor_vector = feno[mejor_individuo]
                self.fitness_values[self.subsector_model].append(mejor_valor)
            self.best_vector[self.subsector_model] = mejor_vector

            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor))
            mse_test = self.get_mse_test(mejor_vector)
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

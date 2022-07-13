import random
import math
import numpy as np
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

from optimization_algorithms import *

'''
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

'''

class RunModel:
    """docstring for CalibrationModel."""

    def __init__(self, df_input_var, country, subsector_model, calib_targets,downstream = False):
        self.df_input_var = df_input_var
        self.country = country
        self.calib_targets = {}
        self.calib_targets[subsector_model] = calib_targets
        self.subsector_model = subsector_model
        self.downstream = downstream
        self.best_vector = {'AFOLU' : None, 'CircularEconomy' : None, 'IPPU': None}

    """
    ---------------------------------
    set_calib_targets method
    ---------------------------------

    Description: The method receive the subsector model to set the calibration targets

    # Inputs:
             * subsector_model              - Subsector model name
             * calib_targets_model          - Calibration targets
    """
    def set_calib_targets(self,subsector_model,calib_targets_model):
        self.calib_targets[subsector_model] = calib_targets_model



    """
    ---------------------------------
    set_best_vector method
    ---------------------------------

    Description: The method receive the subsector model to set the best vector found in
                 the calibration

    # Inputs:
             * subsector_model          - Subsector model name
             * vector                   - vector
    """
    def set_best_vector(self,subsector_model,vector):
        self.best_vector[subsector_model] = vector



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
        df_input_data = self.df_input_var.copy()
        df_input_data = df_input_data.iloc[self.cv_training]

        var_change_over_time = self.df_calib_bounds.query("var_change_over_time==1 and weight_co2==0")["variable"].to_list()
        var_no_change_over_time = self.df_calib_bounds.query("var_change_over_time==0 and weight_co2==0")["variable"].to_list()
        weight_co2 = self.df_calib_bounds.query("weight_co2==1")["variable"].to_list()
        
        index_var_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_change_over_time]
        index_var_no_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_no_change_over_time]
        index_weight_co2 = [list(self.calib_targets[self.subsector_model]).index(i) for i in weight_co2]

        df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

        if list(var_change_over_time):
            df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]
        
        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(df_input_data)
        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)

        if self.subsector_model == "IPPU":
            if not self.downstream:
                #print("\n\tRunning IPPU")
                model_ippu = IPPU(sa.model_attributes)
                df_model_data_project = model_ippu.project(df_input_data)
            else:
                # First run CircularEconomy model
                model_circular_economy = CircularEconomy(sa.model_attributes)
                df_input_data = self.df_input_var.copy()
                df_input_data[self.calib_targets["CircularEconomy"]] = df_input_data[self.calib_targets["CircularEconomy"]] * self.best_vector["CircularEconomy"]
                output_circular_economy = model_circular_economy.project(df_input_data)

                # Build the new df_input_data with the output of CircularEconomy model
                model_ippu = IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                                df_input_data,
                                output_circular_economy,
                                model_ippu.integration_variables
                            )
                # Run IPPU model
                df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

                if list(var_change_over_time):
                    df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]

                df_model_data_project = model_ippu.project(df_input_data)

        return df_model_data_project

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
    * cv_calibration            - Flag that says if the calibration runs cross validation
    * cv_training               - List with the periods of the training set
    * cv_test                   - List with the periods of the test set
    * downstream                - Flag that says if the model is running integrated (i.e. with the output of latest subsector model )

# Output
    * mse_all_period            - Mean Squared Error for all periods
    * mse_training              - Mean Squared Error for training period
    * mse_test                  - Mean Squared Error for test period
    * calib_vector              - List of calibration scalar
'''

class CalibrationModel(RunModel):
    """docstring for CalibrationModel."""


    def __init__(self, df_input_var, country, subsector_model, calib_targets, df_calib_bounds, t_times,
                 df_co2_emissions, cv_calibration = False, cv_training = [], cv_test = [], cv_run = 0, id_mpi = 0,downstream = False,weight_co2_flag = False):
        super(CalibrationModel, self).__init__(df_input_var, country, subsector_model, calib_targets,downstream = False)
        self.df_calib_bounds = df_calib_bounds
        self.df_co2_emissions = df_co2_emissions
        self.cv_calibration = cv_calibration
        self.cv_training = cv_training
        self.cv_test = cv_test
        self.var_co2_emissions_by_sector = {'CircularEconomy' : ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"],
                                            'IPPU': ['emission_co2e_subsector_total_ippu'],
                                            'AFOLU' : ['emission_co2e_subsector_total_agrc','emission_co2e_subsector_total_frst','emission_co2e_subsector_total_lndu','emission_co2e_subsector_total_lvst','emission_co2e_subsector_total_lsmm']}
        self.cv_run = cv_run
        self.id_mpi = id_mpi
        self.fitness_values = {'AFOLU' : [], 'CircularEconomy' : [], 'IPPU' : []}
        self.weight_co2_flag = weight_co2_flag


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

        df_input_data = self.df_input_var.copy()
        df_input_data = df_input_data.iloc[self.cv_training]

        var_change_over_time = self.df_calib_bounds.query("var_change_over_time==1 and weight_co2==0")["variable"].to_list()
        var_no_change_over_time = self.df_calib_bounds.query("var_change_over_time==0 and weight_co2==0")["variable"].to_list()
        weight_co2 = self.df_calib_bounds.query("weight_co2==1")["variable"].to_list()
        
        index_var_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_change_over_time]
        index_var_no_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_no_change_over_time]
        index_weight_co2 = [list(self.calib_targets[self.subsector_model]).index(i) for i in weight_co2]

        df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

        if list(var_change_over_time):
            df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]
        
        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(df_input_data)

        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)

        if self.subsector_model == "IPPU":
            if not self.downstream:
                #print("\n\tRunning IPPU")
                model_ippu = IPPU(sa.model_attributes)
                df_model_data_project = model_ippu.project(df_input_data)
            else:
                # First run CircularEconomy model
                model_circular_economy = CircularEconomy(sa.model_attributes)
                df_input_data = self.df_input_var.copy()
                df_input_data[self.calib_targets["CircularEconomy"]] = df_input_data[self.calib_targets["CircularEconomy"]] * self.best_vector["CircularEconomy"]
                output_circular_economy = model_circular_economy.project(df_input_data)

                # Build the new df_input_data with the output of CircularEconomy model
                model_ippu = IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                                df_input_data,
                                output_circular_economy,
                                model_ippu.integration_variables
                            )
                # Run IPPU model
                df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

                if list(var_change_over_time):
                    df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]

                df_model_data_project = model_ippu.project(df_input_data)

        #out_vars = self.var_co2_emissions_by_sector[self.subsector_model]

        if self.weight_co2_flag:
            out_vars = weight_co2
            model_weight_co2 = np.array(params)[index_weight_co2]/sum(np.array(params)[index_weight_co2])
            model_data_co2e = (df_model_data_project[out_vars]*model_weight_co2).sum(axis=1)
            model_data_co2e = model_data_co2e + df_model_data_project[self.var_co2_emissions_by_sector[self.subsector_model]].sum(axis=1)

        else:
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

        df_input_data = self.df_input_var.copy()
        df_input_data = df_input_data.iloc[self.cv_training]

        var_change_over_time = self.df_calib_bounds.query("var_change_over_time==1 and weight_co2==0")["variable"].to_list()
        var_no_change_over_time = self.df_calib_bounds.query("var_change_over_time==0 and weight_co2==0")["variable"].to_list()
        weight_co2 = self.df_calib_bounds.query("weight_co2==1")["variable"].to_list()
        
        index_var_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_change_over_time]
        index_var_no_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_no_change_over_time]
        index_weight_co2 = [list(self.calib_targets[self.subsector_model]).index(i) for i in weight_co2]

        df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

        if list(var_change_over_time):
            df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]
        
        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(df_input_data)

        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)

        if self.subsector_model == "IPPU":
            if not self.downstream:
                #print("\n\tRunning IPPU")
                model_ippu = IPPU(sa.model_attributes)
                df_model_data_project = model_ippu.project(df_input_data)
            else:
                # First run CircularEconomy model
                model_circular_economy = CircularEconomy(sa.model_attributes)
                df_input_data = self.df_input_var.copy()
                df_input_data[self.calib_targets["CircularEconomy"]] = df_input_data[self.calib_targets["CircularEconomy"]] * self.best_vector["CircularEconomy"]
                output_circular_economy = model_circular_economy.project(df_input_data)

                # Build the new df_input_data with the output of CircularEconomy model
                model_ippu = IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                                df_input_data,
                                output_circular_economy,
                                model_ippu.integration_variables
                            )
                # Run IPPU model
                df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

                if list(var_change_over_time):
                    df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]

                df_model_data_project = model_ippu.project(df_input_data)

        #out_vars = self.var_co2_emissions_by_sector[self.subsector_model]

        if self.weight_co2_flag:
            out_vars = weight_co2
            model_weight_co2 = np.array(params)[index_weight_co2]/sum(np.array(params)[index_weight_co2])
            model_data_co2e = (df_model_data_project[out_vars]*model_weight_co2).sum(axis=1)
            model_data_co2e = model_data_co2e + df_model_data_project[self.var_co2_emissions_by_sector[self.subsector_model]].sum(axis=1)

        else:
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

            n_variables = len(self.calib_targets[self.subsector_model])
            i_sup_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]]
            i_inf_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]
            precision = 8

            binary_genetic = BinaryGenetic(population,n_variables,i_sup_vec,i_inf_vec,precision,maxiter)
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor= binary_genetic.run_optimization(self.f)

            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

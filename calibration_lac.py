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
from scipy.optimize import differential_evolution
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

    def __init__(self, df_input_var, country, subsector_model, calib_targets,df_calib_bounds,all_time_period_input_data = None,downstream = False):
        self.df_input_var = df_input_var
        self.country = country
        self.calib_targets = {}
        self.calib_targets[subsector_model] = calib_targets
        self.subsector_model = subsector_model
        self.df_calib_bounds = df_calib_bounds
        self.downstream = downstream
        self.best_vector = {'AFOLU' : None, 'CircularEconomy' : None, 'IPPU': None}
        self.all_time_period_input_data = all_time_period_input_data
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

        # All time periods input data
        df_input_data = self.all_time_period_input_data.copy()

        agrupa = self.df_calib_bounds.groupby("group")
        group_list = self.df_calib_bounds["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
            index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]
        
        agrupa = self.df_calib_bounds.groupby("norm_group")
        group_list = self.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)
        
        for group in group_list:
            group = int(group)
            if group != 0:
                pij_vars = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]].to_list()
                total_grupo = df_input_data[pij_vars].sum(1)
                for pij_var_ind in pij_vars:
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind]/total_grupo

        self.all_time_period_input_data = df_input_data.copy()


        # CV training input data
        df_input_data = self.df_input_var.copy()
        df_input_data = df_input_data.iloc[self.cv_training]

        agrupa = self.df_calib_bounds.groupby("group")
        group_list = self.df_calib_bounds["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
            index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]

        agrupa = self.df_calib_bounds.groupby("norm_group")
        group_list = self.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)
        
        for group in group_list:
            group = int(group)
            if group != 0:
                pij_vars = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]].to_list()
                total_grupo = df_input_data[pij_vars].sum(1)
                for pij_var_ind in pij_vars:
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind]/total_grupo
                     
        self.imputed_input_data = df_input_data.copy()
        '''
        agrupa = self.df_calib_bounds.groupby("norm_group")
        group_list = self.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group != 0:
                index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                total_pij = df_input_data[index_var_group].sum(1)
                for pij_var in index_var_group:
                    df_input_data[pij_var] =  df_input_data[pij_var]/total_pij        
        '''
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

        item_val_afolu = {}
        item_val_afolu_total_item_fao = {}
        item_val_afolu_total_item_fao_observado = {}

        item_val_afolu_percent_diff = {}

        acumula_total = (self.df_co2_emissions.groupby(["Area_Code","Year"]).sum().reset_index(drop=True).Value/1000).to_numpy()

        for item, vars in self.var_co2_emissions_by_sector[self.subsector_model].items():
            if vars:
                item_val_afolu_total_item_fao[item] = df_model_data_project[vars].sum(1)  
                item_val_afolu_total_item_fao_observado[item] = (self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000
                item_val_afolu[item] = (item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])**2
                #item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) / ((self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000))*100
                item_val_afolu_percent_diff[item] = ((item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])/item_val_afolu_total_item_fao_observado[item])*100

        co2_df = pd.DataFrame(item_val_afolu)
        co2_df_percent_diff = pd.DataFrame(item_val_afolu_percent_diff)
        self.percent_diff = co2_df_percent_diff
        self.error_by_item = co2_df
        self.item_val_afolu_total_item_fao = item_val_afolu_total_item_fao
        co2_df_total = co2_df.sum(1)

        return df_model_data_project

    """
    ---------------------------------
    build_calibrated_data method
    ---------------------------------

    Description: The method recive params and build the calibrated
                 data of specific subsector model.

    # Inputs:
             * params                   - calibration vector
             * df_input_data            - fake data complete

    # Output:
             * df_calibrated_data       - calibrated data
    """

    def build_calibrated_data(self, params,df_input_data):

        agrupa = self.df_calib_bounds.groupby("group")
        group_list = self.df_calib_bounds["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
            index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]

        return df_input_data
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


    def __init__(self, df_input_var, country, subsector_model, calib_targets, df_calib_bounds, all_time_period_input_data,
                 df_co2_emissions, co2_emissions_by_sector = {}, cv_calibration = False, cv_training = [], cv_test = [], cv_run = 0, id_mpi = 0,downstream = False,weight_co2_flag = False, weight_co2 = []):
        super(CalibrationModel, self).__init__(df_input_var, country, subsector_model, calib_targets,df_calib_bounds,all_time_period_input_data,downstream = False)
        self.df_co2_emissions = df_co2_emissions
        self.cv_calibration = cv_calibration
        self.cv_training = cv_training
        self.cv_test = cv_test
        self.var_co2_emissions_by_sector = {'CircularEconomy' : ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"],
                                            'IPPU': ['emission_co2e_subsector_total_ippu'],
                                            'AFOLU' : co2_emissions_by_sector}
        self.cv_run = cv_run
        self.id_mpi = id_mpi
        self.fitness_values = {'AFOLU' : [], 'CircularEconomy' : [], 'IPPU' : []}
        self.weight_co2_flag = weight_co2_flag
        self.weight_co2 = np.array(weight_co2)
        self.item_val_afolu_total_item_fao = None

    def get_calib_var_group(self,grupo): 
        return self.df_calib_bounds.query("group =={}".format(grupo))["variable"].to_list()

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

        '''
        var_change_over_time = self.df_calib_bounds.query("var_change_over_time==1")["variable"].to_list()
        var_no_change_over_time = self.df_calib_bounds.query("var_change_over_time==0")["variable"].to_list()
        
        index_var_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_change_over_time]
        index_var_no_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_no_change_over_time]

        df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

        if list(var_change_over_time):
            df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]
        '''
        agrupa = self.df_calib_bounds.groupby("group")
        group_list = self.df_calib_bounds["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
            index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]

        agrupa = self.df_calib_bounds.groupby("norm_group")
        group_list = self.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)
        
        for group in group_list:
            group = int(group)
            if group != 0:
                pij_vars = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]].to_list()
                total_grupo = df_input_data[pij_vars].sum(1)
                for pij_var_ind in pij_vars:
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind]/total_grupo
                     
        '''
        agrupa = self.df_calib_bounds.groupby("norm_group")
        group_list = self.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group != 0:
                index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                total_pij = df_input_data[index_var_group].sum(1)
                for pij_var in index_var_group:
                    df_input_data[pij_var] =  df_input_data[pij_var]/total_pij        
        '''
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

        out_vars = self.var_co2_emissions_by_sector[self.subsector_model]

        '''
        if self.weight_co2_flag:
            model_data_co2e = (df_model_data_project[out_vars]*model_weight_co2).sum(axis=1)
            model_data_co2e = model_data_co2e + df_model_data_project[self.var_co2_emissions_by_sector[self.subsector_model]].sum(axis=1)

        else:
            model_data_co2e = df_model_data_project[out_vars].sum(axis=1)
        '''

        #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
        #trend = self.df_co2_emissions.value
        #trend = [i/1000 for i in trend]

        item_val_afolu = {}
        item_val_afolu_total_item_fao = {}
        item_val_afolu_total_item_fao_observado = {}
        item_val_afolu_percent_diff = {}
        acumula_total = (self.df_co2_emissions.groupby(["Area_Code","Year"]).sum().reset_index(drop=True).Value/1000).to_numpy()

        for item, vars in self.var_co2_emissions_by_sector[self.subsector_model].items():
            if vars:
                item_val_afolu_total_item_fao[item] = df_model_data_project[vars].sum(1)  
                item_val_afolu_total_item_fao_observado[item] = (self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000
                item_val_afolu[item] = (item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])**2
                #item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) / ((self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000))*100
                item_val_afolu_percent_diff[item] = ((item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])/item_val_afolu_total_item_fao_observado[item])*100

        co2_df = pd.DataFrame(item_val_afolu)
        co2_df_percent_diff = pd.DataFrame(item_val_afolu_percent_diff)
        self.percent_diff = co2_df_percent_diff
        self.error_by_item = co2_df
        self.item_val_afolu_total_item_fao = item_val_afolu_total_item_fao
        co2_df_total = co2_df.sum(1)

        if self.cv_calibration:
            co2_df_total = np.array([co2_df_total[i] for i in self.cv_training])
            output = np.mean(co2_df_total)
        else:
            output = np.mean(co2_df_total)

            if any((np.mean(self.percent_diff[["5058","6750"]]).to_numpy()>30).flatten()):
                return output + 10000000
            else:

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

        '''
        var_change_over_time = self.df_calib_bounds.query("var_change_over_time==1")["variable"].to_list()
        var_no_change_over_time = self.df_calib_bounds.query("var_change_over_time==0")["variable"].to_list()
        
        index_var_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_change_over_time]
        index_var_no_change_over_time = [list(self.calib_targets[self.subsector_model]).index(i) for i in var_no_change_over_time]

        df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

        if list(var_change_over_time):
            df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]
        '''

        agrupa = self.df_calib_bounds.groupby("group")
        group_list = self.df_calib_bounds["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
            index_var_group = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]

        agrupa = self.df_calib_bounds.groupby("norm_group")
        group_list = self.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)
        
        for group in group_list:
            group = int(group)
            if group != 0:
                pij_vars = self.df_calib_bounds["variable"].iloc[agrupa.groups[group]].to_list()
                total_grupo = df_input_data[pij_vars].sum(1)
                for pij_var_ind in pij_vars:
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind]/total_grupo
                     
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

        out_vars = self.var_co2_emissions_by_sector[self.subsector_model]


        '''
        if self.weight_co2_flag:
            model_data_co2e = (df_model_data_project[out_vars]*model_weight_co2).sum(axis=1)
            model_data_co2e = model_data_co2e + df_model_data_project[self.var_co2_emissions_by_sector[self.subsector_model]].sum(axis=1)

        else:
            model_data_co2e = df_model_data_project[out_vars].sum(axis=1)
        '''

        #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
        #trend = self.df_co2_emissions.value
        #trend = [i/1000 for i in trend]

        item_val_afolu = {}
        item_val_afolu_percent_diff = {}
        acumula_total = (self.df_co2_emissions.groupby(["Area_Code","Year"]).sum().reset_index(drop=True).Value/1000).to_numpy()

        for item, vars in self.var_co2_emissions_by_sector[self.subsector_model].items():
            if vars:
                item_val_afolu[item] = (df_model_data_project[vars].sum(1) - (self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000)**2
                #item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) / ((self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000))*100
                item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) - (self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000) / (self.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value/1000)

        co2_df = pd.DataFrame(item_val_afolu)
        co2_df_percent_diff = pd.DataFrame(item_val_afolu_percent_diff)
        self.percent_diff = co2_df_percent_diff
        self.error_by_item = co2_df
        co2_df_total = co2_df.sum(1)

        if self.cv_calibration:
            co2_df_total = np.array([co2_df_total[i] for i in self.cv_training])
            output = np.mean(co2_df_total)
        else:
            output = np.mean(co2_df_total)

            if any((np.mean(self.percent_diff[["5058","6750"]]).to_numpy()>30).flatten()):
                return output + 10000000
            else:

                return output


    def run_calibration(self,optimization_method,population, maxiter):
        '''
        ------------------------------------------

        run_calibration method

        -------------------------------------------

        # Inputs:
            * optimization_method       - Optimization method. Args: genetic_binary, differential_evolution,differential_evolution_parallel, pso


        # Output
            * mse_all_period            - Mean Squared Error for all periods
            * mse_training              - Mean Squared Error for training period
            * mse_test                  - Mean Squared Error for test period
            * calib_vector              - List of calibration scalar
        '''

        if optimization_method == "genetic_binary":
            # Optimize the function
            print("------------------------------------------------")
            print("         GENETIC BINARY      ")
            print("------------------------------------------------")
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

        if optimization_method == "differential_evolution":
            print("------------------------------------------------")
            print("         DIFERENTIAL EVOLUTION      ")
            print("------------------------------------------------")

            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            # 1 - Define the upper and lower bounds of the search space
            n_dim =  len(self.calib_targets[self.subsector_model])              # Number of dimensions of the problem
            lb = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]  # lower bound for the search space
            ub =  [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]] # upper bound for the search space
            lb = np.array(lb)
            ub = np.array(ub)

            # 2 - Define the parameters for the optimization
            max_iters = maxiter  # maximum number of iterations

            # 3 - Parameters for the algorithm
            pc = 0.7 # crossover probability
            pop_size = population   # number of individuals in the population

            # 4 - Define the cost function
            f_cost = self.f

            # 5 - Run the DE algorithm
            # call DE
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor = DE(f_cost,pop_size, max_iters,pc, lb, ub, step_size = 0.8)
            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor[0]))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

        if optimization_method == "differential_evolution_parallel":
            
            print("------------------------------------------------")
            print("         PARALLEL DIFERENTIAL EVOLUTION      ")
            print("------------------------------------------------")

            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            # 1 - Define the upper and lower bounds of the search space
            n_dim =  len(self.calib_targets[self.subsector_model])              # Number of dimensions of the problem
            lb = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]  # lower bound for the search space
            ub =  [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]] # upper bound for the search space
            lb = np.array(lb)
            ub = np.array(ub)

            # 2 - Define the parameters for the optimization
            max_iters = maxiter  # maximum number of iterations

            # 3 - Parameters for the algorithm
            pc = 0.7 # crossover probability
            pop_size = population   # number of individuals in the population

            # 4 - Define the cost function
            f_cost = self.f

            # 5 - Run the DE algorithm
            # call DE
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor = DE_par(f_cost,pop_size, max_iters,pc, lb, ub, step_size = 0.4)
            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor[0]))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

        if optimization_method == "pso":
            
            print("------------------------------------------------")
            print("         PARTICLE SWARM OPTIMIZATION      ")
            print("------------------------------------------------")

            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            # 1 - Define the upper and lower bounds of the search space
            n_dim =  len(self.calib_targets[self.subsector_model])              # Number of dimensions of the problem
            lb = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]  # lower bound for the search space
            ub =  [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]] # upper bound for the search space
            lb = np.array(lb)
            ub = np.array(ub)

            # 2 - Define the parameters for the optimization
            max_iters = maxiter  # maximum number of iterations

            # 3 - Parameters for the algorithm
            pop_size = population   # number of individuals in the population
            # Cognitive scaling parameter
            α = 0.8
            # Social scaling parameter
            β = 0.8
            # velocity inertia
            w = 0.5
            # minimum value for the velocity inertia
            w_min = 0.4
            # maximum value for the velocity inertia
            w_max = 0.9

            # 4 - Define the cost function
            f_cost = self.f

            # 5 - Run the PSO algorithm
            # call DE
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor = PSO(f_cost,pop_size, max_iters, lb, ub,α,β,w,w_max,w_min)
            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

 
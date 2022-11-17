import functools
import numpy as np
import pandas as pd
import math

# Load LAC-Decarbonization source
import sys
import os
cwd = os.getcwd()

sys.path.append(cwd + '/lac_decarbonization/python')

from data_functions_mix_lndu_transitions_from_inferred_bounds import MixedLNDUTransitionFromBounds

"""
Decorators for build data per sector
"""

def data_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,params):

        df_input_data = calibration.df_input_var.copy()
        df_input_data = df_input_data.iloc[calibration.cv_training]

        agrupa = calibration.df_calib_bounds.groupby("group")
        group_list = calibration.df_calib_bounds["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
                #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))
            index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]
            #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))

        agrupa = calibration.df_calib_bounds.groupby("norm_group")
        group_list = calibration.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)
        
        for group in group_list:
            group = int(group)
            if group != 0:
                pij_vars = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]].to_list()
                total_grupo = df_input_data[pij_vars].sum(1)
                for pij_var_ind in pij_vars:
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind]/total_grupo
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind].apply(lambda x : round(x, calibration.precition)) 
                     

        # Do something after
        return df_input_data
    return wrapper_decorator

def data_matrix_pij_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,params):

        df_input_data = calibration.df_input_var.copy()
        df_input_data = df_input_data.iloc[calibration.cv_training]

        agrupa = calibration.df_calib_bounds.query("group!=999").groupby("group")
        group_list = calibration.df_calib_bounds.query("group!=999")["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)-2]
                #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))
            index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups-2]
            #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))

        mixer = MixedLNDUTransitionFromBounds(eps = 0.0001)# eps is a correction threshold for transition matrices
        prop_pij = mixer.mix_transitions(params[-1],calibration.country)
        prop_pij = prop_pij.query(f"year> {calibration.year_init-3}").reset_index(drop=True)
        df_input_data[prop_pij.columns[1:]] = prop_pij[prop_pij.columns[1:]]
        # Do something after
        return df_input_data
    return wrapper_decorator


"""
Decorators for get_output_data
"""
def get_output_data_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):
        item_val_afolu = {}
        item_val_afolu_total_item_fao = {}
        item_val_afolu_total_item_fao_observado = {}

        item_val_afolu_percent_diff = {}

        acumula_total = (calibration.df_co2_emissions.groupby(["Area_Code","Year"]).sum().reset_index(drop=True).Value/1000).to_numpy()

        for item, vars in calibration.var_co2_emissions_by_sector[calibration.subsector_model].items():
            if vars:
                item_val_afolu_total_item_fao[item] = df_model_data_project[vars].sum(1)  
                item_val_afolu_total_item_fao_observado[item] = (calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000
                item_val_afolu[item] = (item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])**2
                #item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) / ((calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000))*100
                item_val_afolu_percent_diff[item] = ((item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])/item_val_afolu_total_item_fao_observado[item])*100

        co2_df = pd.DataFrame(item_val_afolu)
        co2_df_percent_diff = pd.DataFrame(item_val_afolu_percent_diff)
        calibration.percent_diff = co2_df_percent_diff
        calibration.error_by_item = co2_df
        calibration.item_val_afolu_total_item_fao = item_val_afolu_total_item_fao
        co2_df_total = co2_df.sum(1)
    return wrapper_decorator

"""
Decorators for performance metrics
"""


def performance_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):
        #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
        #trend = calibration.df_co2_emissions.value
        #trend = [i/1000 for i in trend]

        item_val_afolu = {}
        item_val_afolu_total_item_fao = {}
        item_val_afolu_total_item_fao_observado = {}
        item_val_afolu_percent_diff = {}
        acumula_total = (calibration.df_co2_emissions.groupby(["Area_Code","Year"]).sum().reset_index(drop=True).Value/1000).to_numpy()

        for item, vars in calibration.var_co2_emissions_by_sector[calibration.subsector_model].items():
            if vars:
                item_val_afolu_total_item_fao[item] = df_model_data_project[vars].sum(1)  
                item_val_afolu_total_item_fao_observado[item] = (calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000
                item_val_afolu[item] = (item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])**2
                #item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) / ((calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000))*100
                item_val_afolu_percent_diff[item] = ((item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])/item_val_afolu_total_item_fao_observado[item])*100

        co2_df = pd.DataFrame(item_val_afolu)
        co2_df_percent_diff = pd.DataFrame(item_val_afolu_percent_diff)
        calibration.percent_diff = co2_df_percent_diff
        calibration.error_by_item = co2_df
        calibration.item_val_afolu_total_item_fao = item_val_afolu_total_item_fao
        co2_df_total = co2_df.sum(1)

        co2_df_observado = pd.DataFrame(item_val_afolu_total_item_fao_observado)

        ponderadores = (co2_df_observado.mean().abs()/co2_df_observado.mean().abs().sum()).apply(math.exp)   
        co2_df_total = (ponderadores*co2_df).sum(1)

        if calibration.cv_calibration:
            """
            co2_df_total = np.array([co2_df_total[i] for i in calibration.cv_training])
            output = np.mean(co2_df_total)
            """
            co2_df_total = [co2_df_total[i] for i in calibration.cv_training]
            output = np.sum(co2_df_total)

        else:
            output = np.sum(co2_df_total)
            """
            if any((np.mean(calibration.percent_diff[["5058","6750"]]).to_numpy()>30).flatten()):
                return output + 10000000
            else:
            """

        # Do something after
        return output
    return wrapper_decorator

"""
Decorators for performance metrics in test set
"""

def performance_test_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):
        #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
        #trend = calibration.df_co2_emissions.value
        #trend = [i/1000 for i in trend]

        item_val_afolu = {}
        item_val_afolu_percent_diff = {}
        item_val_afolu_total_item_fao_observado = {}

        acumula_total = (calibration.df_co2_emissions.groupby(["Area_Code","Year"]).sum().reset_index(drop=True).Value/1000).to_numpy()

        for item, vars in calibration.var_co2_emissions_by_sector[calibration.subsector_model].items():
            if vars:
                item_val_afolu[item] = (df_model_data_project[vars].sum(1) - (calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000)**2
                item_val_afolu_total_item_fao_observado[item] = (calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000
                #item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) / ((calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000))*100
                item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) - (calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000) / (calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value/1000)

        co2_df = pd.DataFrame(item_val_afolu)
        co2_df_percent_diff = pd.DataFrame(item_val_afolu_percent_diff)
        calibration.percent_diff = co2_df_percent_diff
        calibration.error_by_item = co2_df
        co2_df_total = co2_df.sum(1)

        co2_df_observado = pd.DataFrame(item_val_afolu_total_item_fao_observado)

        ponderadores = (co2_df_observado.mean().abs()/co2_df_observado.mean().abs().sum()).apply(math.exp)

        co2_df_total = (ponderadores*co2_df).sum(1)

        if calibration.cv_calibration:
            """
            co2_df_total = np.array([co2_df_total[i] for i in calibration.cv_training])
            output = np.mean(co2_df_total)
            """
            co2_df_total = [co2_df_total[i] for i in calibration.cv_training]
            output = np.sum(co2_df_total)

        else:
            output = np.sum(co2_df_total)
            """
            if any((np.mean(calibration.percent_diff[["5058","6750"]]).to_numpy()>30).flatten()):
                return output + 10000000
            else:
            """
        # Do something after
        return output
    return wrapper_decorator


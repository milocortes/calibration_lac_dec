import functools
import numpy as np

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
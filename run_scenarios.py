import os
import sys

ACTUAL_PATH = os.getcwd()

LAC_PATH = '/home/milo/Documents/egap/escenarios_SSP/lac_decarbonization'

sys.path.append(os.path.join(LAC_PATH, 'python'))


import logging
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import temp as tmp
import sisepuede as ssp
import sqlalchemy
import sql_utilities as sqlutil
import support_functions as sf
from typing import *
import warnings
warnings.filterwarnings("ignore")



##
def _setup_logger(namespace: str, fn_out: Union[str, None] = None) -> None:
    global logger
    
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    # configure
    if fn_out is not None:
        logging.basicConfig(
            filename = fn_out,
            filemode = "w",
            format = format_str,
            level = logging.DEBUG
        )
    else:
        logging.basicConfig(
            format = format_str,
            level = logging.DEBUG
        )
        
    logger = logging.getLogger(namespace)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter(format_str)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger

_setup_logger(__name__, os.path.join(os.getcwd(), "log_sisepuede.log"))

import sisepuede_file_structure as sfs
file_struct = sfs.SISEPUEDEFileStructure()

# Initialize the SISEPUEDE class to get started running some models
regions =[
    "mexico"
]

sisepuede = ssp.SISEPUEDE(
    "calibrated", 
    id_str = "sisepuede_run_2023-06-20T03:12:44.178024",#id_str = "sisepuede_run_2023-06-13T00:55:37.051768",
    logger = logger,
    regions = regions
)


# project across 2 futures for 1 design and, notably, *all* strategies (no filtering)
dict_filt = {
    "future_id": list(range(10)),
    "design_id": [0], 
    "strategy_id": [0]
}

primary_keys_out = sisepuede.project_scenarios(
    dict_filt,
    chunk_size = 2,
    regions = regions
)

# Print successfully completed primary keys
for k in primary_keys_out:
    print(k)

df_out_all = sisepuede.read_output(None)
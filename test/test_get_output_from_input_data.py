import os

os.environ['LAC_PATH'] = '/home/milo/Documents/julia/lac_decarbonization'

from sisepuede_calibration.calibration_lac import ModelOutputData

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/jcsyme/sisepuede/main/ref/fake_data/fake_data_complete.csv")

## Test AFOLU
ModelOutputData.get_output_input_data_AFOLU(df, False)

## Test CircularEconomy
ModelOutputData.get_output_input_data_CircularEconomy(df, False)
ModelOutputData.get_output_input_data_CircularEconomy(df, True)

## Test IPPU
ModelOutputData.get_output_input_data_IPPU(df, False)
ModelOutputData.get_output_input_data_IPPU(df, True)

## Test NonElectricEnergy
ModelOutputData.get_output_input_data_NonElectricEnergy(df, True)

## Test ElectricEnergy
ModelOutputData.get_output_input_data_ElectricEnergy(df, True)

## Test AllEnergy
ModelOutputData.get_output_input_data_AllEnergy(df, True)
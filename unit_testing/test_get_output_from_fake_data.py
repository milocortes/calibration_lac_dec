import os

os.environ['LAC_PATH'] = '/home/milo/Documents/julia/lac_decarbonization'

from sisepuede_calibration.calibration_lac import TestFakeData

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/egobiernoytp/lac_decarbonization/main/ref/fake_data/fake_data_complete.csv")

## Test AFOLU
TestFakeData.get_output_fake_data_AFOLU(df, False)

## Test
TestFakeData.get_output_fake_data_CircularEconomy(df, False)
TestFakeData.get_output_fake_data_CircularEconomy(df, True)

## Test IPPU
TestFakeData.get_output_fake_data_IPPU(df, False)
TestFakeData.get_output_fake_data_IPPU(df, True)

## Test NonElectricEnergy
TestFakeData.get_output_fake_data_NonElectricEnergy(df, True)

## Test ElectricEnergy
TestFakeData.get_output_fake_data_ElectricEnergy(df, True)

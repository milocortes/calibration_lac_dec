#!/usr/bin/env bash

filename="countries_arguments_afolu.txt"

while IFS=, read -r country lower upper
do
  echo "RUN $country CALIBRATION"
  </dev/null python calibration_models_afolu_no_mpi.py "$country"
done < "$filename"

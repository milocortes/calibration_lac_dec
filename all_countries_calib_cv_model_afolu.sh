#!/usr/bin/env bash

filename="countries_arguments_afolu.txt"

while IFS=, read -r country lower upper
do
  echo "RUN $country CALIBRATION"
  </dev/null mpiexec --oversubscribe -np 2 python calibration_models_afolu.py "$country"
done < "$filename"

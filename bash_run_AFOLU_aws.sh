#!/usr/bin/env bash

filename="countries_arguments_afolu.txt"

while IFS=, read -r country lower upper
do
  echo "RUN $country CALIBRATION"
  #</dev/null python run_AFOLU_aws.py "$country"
  </dev/null python test_precision_afolu.py "$country"
done < "$filename"

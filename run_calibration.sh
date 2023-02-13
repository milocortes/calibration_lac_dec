#!/bin/bash

paises=(CHL COL CRI DOM ECU GTM HND  JAM MEX NIC PAN PER PRY SLV URY)

for i in "${paises[@]}"
do
	echo "RUN $i CALIBRATION"
	python src/calibration_routine_country.py $i
done

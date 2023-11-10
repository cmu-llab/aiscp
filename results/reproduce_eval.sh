#!/bin/bash

> results/reproduce.txt
echo "min, argmin, mean, stdev" >> results/reproduce.txt

for experiment in dirpath step2_mod_1L step2_mod_8res step3_mod_4L_kOOV step1_kOOV step2_mod_1L_kOOV step2_mod_8res_kOOV step3_mod_8res_kOOV step2_mod_16res step2_mod_4L step3_mod_16res_kOOV step2_mod_16res_kOOV step2_mod_4L_kOOV step3_mod_1L_kOOV
do
    echo $experiment >> results/reproduce.txt
    gqd_array=()
    for i in $(seq 1 10)
    do
        echo "run $i" >> results/reproduce.txt
        if [[ $experiment = step3* ]]
        then
            gqd=$(./evaluation/evalstep3.sh results/res10000_run$i/$experiment/R_diwest-* | grep GQD)
            gqd="${gqd##* }"
            gqd_array+=($gqd)
        else
            gqd=$(./evaluation/evaluate.sh results/res10000_run$i/$experiment/R_diwest-* | grep GQD)
            gqd="${gqd##* }"
            gqd_array+=($gqd)
        fi
    done

    printf -v joined '%s,' "${gqd_array[@]}"
    gqd_array="${joined%,}"
    python -c "import numpy as np; print( np.min([ $gqd_array ]), np.argmin([ $gqd_array ]), np.mean([ $gqd_array ]), np.std([ $gqd_array ], ddof=1) )" >> results/reproduce.txt

    echo "" >> results/reproduce.txt
done

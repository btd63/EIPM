#!/bin/bash
set -e

PYTHON=python
SCRIPT=my_dgp.py

# ----------------------------
# Fixed settings
# ----------------------------
N_EVAL=10000
N_RPT=100
SEED=42

N_TRAIN_LIST=(250 500 1000 2000)
SCENARIOS=(linear nonlinear)
PI_0S=(0.0 0.2 0.8)
echo "=== Running all DGP combinations ==="

#######################################
# 1. d_X = 5
#######################################
for N_TRAIN in "${N_TRAIN_LIST[@]}"
do
    for SCENARIO in "${SCENARIOS[@]}"
    do
        for PI_0 in "${PI_0S[@]}"
        do
            echo "[d_X=5 | ${SCENARIO} | n_train=${N_TRAIN} | pi_0=${PI_0}]"
            ${PYTHON} ${SCRIPT} \
                --scenario ${SCENARIO} \
                --d_X 5 \
                --n_train ${N_TRAIN} \
                --n_eval ${N_EVAL} \
                --n_rpt ${N_RPT} \
                --pi_0 ${PI_0} \
                --seed ${SEED} \
                --treatment_k 5 \
                --outcome_k 5
        done
    done
done

#######################################
# 2. d_X = 50
#######################################
for N_TRAIN in "${N_TRAIN_LIST[@]}"
do
    for SCENARIO in "${SCENARIOS[@]}"
    do
        for TK in 50 5
        do
            for OK in 50 5
            do
                for PI_0 in "${PI_0S[@]}"
                do
                    echo "[d_X=50 | ${SCENARIO} | n_train=${N_TRAIN} | pi_0=${PI_0} | T_k=${TK} | Y_k=${OK}]"
                    ${PYTHON} ${SCRIPT} \
                        --scenario ${SCENARIO} \
                        --d_X 50 \
                        --n_train ${N_TRAIN} \
                        --n_eval ${N_EVAL} \
                        --n_rpt ${N_RPT} \
                        --pi_0 ${PI_0} \
                        --seed ${SEED} \
                        --treatment_k ${TK} \
                        --outcome_k ${OK}
                done
            done
        done
    done
done

echo "=== All DGP jobs completed successfully ==="

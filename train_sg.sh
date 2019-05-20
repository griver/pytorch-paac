#!/usr/bin/env bash

topologies=('erdos_renyi-n100' 'watts_strogatz-n100') #'torus-10x10' 'barabasi_albert-n100'
graph_ids=(1 2 3 4 5 6 7 8 9 10) #(1 2 3 4 5 ) ids are done for the latst period
update_period=(1 2 5 10 20 50 100 500) # (1 2 5 10 20 50 100 500) periods are done

living_reward="-0.01"

for id in "${graph_ids[@]}"; do
    for top in "${topologies[@]}"; do
        for per in "${update_period[@]}"; do

            echo "##################################################"
            echo "Train a2c-lstm on ${top}-${id} with period=${per}"

            python3 train_stoch_graph.py \
                -g "${top}-${id}" -d cuda -w 8 -n 32 \
                --max-global-steps 5000000 -lra 7000000 -r 10 \
                -sf "pretrained/stoch_graphs/period_${per}/a2c-${top}-${id}/" \
                -p $per  --max-episode-steps 300 --algo a2c \
                --living_r $living_reward
            sleep 5s

        done
    done
done
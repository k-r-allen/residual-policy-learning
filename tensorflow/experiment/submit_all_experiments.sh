#!/bin/bash
envs=("FetchPush-v1" "FetchPickAndPlace-v1" "FetchPushHighFriction-v0" "TwoFrameHookNoisy-v0" "ComplexHookTrain-v0")
residuals=("ResidualMPCPush-v0" "ResidualFetchPickAndPlace-v0" "ResidualFetchPush-v0" "TwoFrameResidualHookNoisy-v0" "ResidualComplexHookTrain-v0")

expertexplore=("MPCPush-v0" "FetchPickAndPlace-v1" "FetchPushHighFriction-v0" "TwoFrameHookNoisy-v0" "ComplexHookTrain-v0")
eps=(0.5 0.5 0.5 0.6 0.6)
alpha=(0.8 0.8 0.8 0.8 0.8)
configs=("push.json" "pickandplace.json" "push.json" "hook.json" "hook.json")
cpus=(19 19 19 1 1)
nepochs=(50 50 50 300 300)
seeds=(0 1 2 3 4)

start=$1
num_experiments=$2
experiment_counter=0

for j in ${!seeds[@]}; do
    #Train expert-explore
    for i in ${!expertexplore[@]}; do
      if (( experiment_counter >= start && experiment_counter < (start + num_experiments) )); then
        python train_staged.py --env ${expertexplore[$i]} --n_epochs ${nepochs[$i]} --num_cpu ${cpus[$i]} --config_path=configs/${configs[$i]} --logdir ./logs/seed${seeds[$j]}/${expertexplore[$i]}_expertexplore --seed ${seeds[$j]} --random_eps=${eps[$i]} --controller_prop=${alpha[$i]}
      fi
      ((experiment_counter++))
    done

    #Train from scratch
    for i in ${!envs[@]}; do
      if (( experiment_counter >= start && experiment_counter < (start + num_experiments) )); then
        python train_staged.py --env ${envs[$i]} --n_epochs ${nepochs[$i]} --num_cpu ${cpus[$i]} --config_path=configs/${configs[$i]} --logdir ./logs/seed${seeds[$j]}/${envs[$i]} --seed ${seeds[$j]}
      fi
      ((experiment_counter++))
    done

    #Train residuals
    for i in ${!residuals[@]}; do
      if (( experiment_counter >= start && experiment_counter < (start + num_experiments) )); then
        python train_staged.py --env ${residuals[$i]} --n_epochs ${nepochs[$i]} --num_cpu ${cpus[$i]} --config_path=configs/${configs[$i]} --logdir ./logs/seed${seeds[$j]}/${residuals[$i]} --seed ${seeds[$j]}
      fi
      ((experiment_counter++))
    done
done

#!/usr/bin/env bash
pwd; hostname; date


remote_dev_env="local"
result_name="novel_cds_run"
result_folder="exp_one"

if [[ "$remote_dev_env" == "local" ]]; then
  echo "running on my computer"
  VENV_PATH="./.venv"
  data_path='./databsd/'
  base_dir='./results/'
  python_script="./main.py"
else
  echo "Error: Invalid remote_dev_env value" >&2
  exit 1
fi

path_results="${base_dir}${result_folder}/${result_name}"
echo "Full path: ${path_results}"


model_name='cds' # scn, ainet, ssm, cds
pipeline='t_sep_v' #tv, t_sep_v
train_size=208
val_size=320
device='cuda'

epochs=736 # (1500 for scn baseline, 2500 scn novel,736 cds baseline, 1500 cds novel, ainet baseline tsv 3000, ainet baseline tv 1000 (1500 for novel)
# ssm baseline 750, ssm novel 1125
# AInet novel only includes the new losses at the tv stage.
# ssm we only did the second stage, weights_trained="tv_only" need to ensure the weights are loaded in models/ssm/
batch_size=8 # 8 for scn/ssm/cds and 16 for ainet
num_batches=1.0 # entire dataset
num_workers=6
learning_rate=5e-4 # SCN 5e-5, SSM and CDS 5e-4, AInet 8e-5
#weight_decay=4e-4 # for tv
weight_decay=0.0


# MAKE SURE TO SWITCH THE TRAINING AND VAL STEP CODE in MyModules.py
#loss_weights='{"recon":1.0,"compact":0.0001875,"contrastive":0.0,"lap":0.0}' # baseline
loss_weights='{"recon":1.0,"compact":0.0001875,"contrastive":0.001,"lap":360.0}' # novel
loss_name="cdspixel_novel"

# trained weight path usually load weights here from tsv for tv
weights_trained="/path/to/weights/weights.ckpt"

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi


# If only single gpu then CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=1 python "$python_script" --model_name $model_name --pipeline $pipeline --data_dir $data_path \
--result_path $path_results --remote $remote_dev_env --weight_decay $weight_decay --lr $learning_rate --device $device --num_workers $num_workers \
--batch_size $batch_size --num_batches $num_batches --epochs $epochs --train_size $train_size --val_size $val_size \
--loss_weights "$loss_weights" --loss_name $loss_name #--weights_to_load $weights_trained # Uncomment for TV


#nohup ./submit_job.sh > log.txt 2>&1 &
#to kill a nohup job
#jobs -l
#select the correct job pid and kill <pid>, you will probably need to kill the jobs on the gpu, nvidia-smi kill -9 <pid>

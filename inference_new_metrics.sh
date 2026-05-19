#!/usr/bin/env bash
pwd; hostname; date

VENV_PATH="./.venv"

# Check if already in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

model_prefix="cds"
pretrained_weights=(
"./weights/models_lap/cds_lap.ckpt"
)


echo "Array length: ${#pretrained_weights[@]}"

datasets="bsd"
index=0
plotlist=(0)  # No plotting

# Check lengths
if [ ${#pretrained_weights[@]} -ne ${#plotlist[@]} ]; then
  echo "Error: Arrays have different lengths!"
  echo "pretrained_weights has ${#pretrained_weights[@]} elements"
  echo "plotting has ${#plotlist[@]} elements"
  exit 1
fi

for weight_path in "${pretrained_weights[@]}";
 do

  echo "Model name: $model_prefix"
  weight_path="${pretrained_weights[$index]}"
  plot_flag="${plotlist[$index]}"
  echo "plot flag: $plot_flag"

  for data in $datasets
  do
    foldername="${data}_new_metrics_noEC_${model_prefix}"
    echo "$foldername"

    CUDA_VISIBLE_DEVICES=0 python ./eval_spixel/infer_newmetrics.py --folder_name $foldername \
  --pretrained_weight_path $weight_path --model_name $model_prefix --data_name $data --plot_flag $plot_flag

  echo "Exit code: $?"

  done
  echo "Incrementing index from $index"
  ((index++))
  echo "New index: $index"
echo "Finished $foldername"
  done

echo "All Done"
#nohup ./custom_infer.sh > log.txt 2>&1 &
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
"./weights/models_lap/cds_lap.ckpt"
)

#datasets="voc nyu"
datasets="bsd"
index=0


enforce_con_bools=(
0 1
)


count=0
for val in "${enforce_con_bools[@]}"; do
  ((count += val))
done
echo "Number of 1s: $count"


# Check lengths
if [ ${#pretrained_weights[@]} -ne ${#enforce_con_bools[@]} ]; then
  echo "Error: Arrays have different lengths!"
  echo "pretrained_weights has ${#pretrained_weights[@]} elements"
  echo "enforce_con_bools has ${#enforce_con_bools[@]} elements"
  exit 1
fi

# Continue with your script if lengths match
echo "Arrays match: ${#pretrained_weights[@]} elements each"


for weight_path in "${pretrained_weights[@]}";
 do
  weight_path="${pretrained_weights[$index]}"
  enforce_con_bool="${enforce_con_bools[$index]}"
  if [ "$enforce_con_bool" -eq 1 ]; then
    ec_suffix="withEC"
  else
    ec_suffix="noEC"
  fi

  for data in $datasets
  do
    foldername="${data}_testinfer_lap_${model_prefix}_${ec_suffix}"  # Append the EC suffix
    echo "$foldername"

  # folders for final result location
  src_dir=./results/test_set/${foldername}
  dst_dir=./results/test_set/plot_test_results/${foldername}


  CUDA_VISIBLE_DEVICES=0 python ./eval_spixel/infer.py --folder_name $foldername \
  --pretrained_weight_path $weight_path --enforce_con $enforce_con_bool --model_name $model_prefix --data_name $data

  # this is script to generate the per image statistics using the benchmark code. the command has to be ran from
  # ./superpixel-benchmark/examples

  # Save current directory
  ORIG_DIR=$(pwd)

  # Change to the target directory
  cd ./superpixel-benchmark/examples || exit 1


  if [ "$data" = "bsd" ]; then
      IMG_PATH=../data/BSDS500/images/test
      GT_PATH=../data/BSDS500/csv_groundTruth/test
      SUPERPIXELS=("54" "96" "150" "216" "294" "384" "486" "600" "726" "864" "1014" "1176" "1350")
  elif [ "$data" = "nyu" ]; then
      IMG_PATH=../data/NYUV2/img
      GT_PATH=../data/NYUV2/label_csv
      SUPERPIXELS=("300" "432" "588" "768" "972" "1200" "1452" "1728" "2028" "2352")

  elif [ "$data" = "voc" ]; then
      IMG_PATH=../data/VOC/img
      GT_PATH=../data/VOC/label_csv
      SUPERPIXELS=("81" "144" "225" "324" "441" "576" "729" "900" "1089" "1296")
  else
      echo "Unknown dataset: $data"
      exit 1
  fi

  echo "IMG_PATH: $IMG_PATH"
  echo "GT_PATH: $GT_PATH"
  sleep 2
  for SUPERPIXEL in "${SUPERPIXELS[@]}"
  do
     echo $SUPERPIXEL
         ../bin/eval_summary_cli ../../results/test_set/${foldername}/SPixelNet_nSpixel_${SUPERPIXEL}/map_csv $IMG_PATH $GT_PATH
  done
  #return to the original directory
  cd "$ORIG_DIR"
  sleep 2 #sometimes the files do not update quickly enough and it does not find the files

  python ./eval_spixel/copy_results.py --src_dir $src_dir --dst_dir $dst_dir --data_name $data
  done
    echo "Incrementing index from $index"
  ((index++))
  echo "New index: $index"
echo "Finished $foldername"
done
echo "All Done"
#nohup ./inference.sh > log.txt 2>&1 &
#!/usr/bin/env bash

echo pwd: $(pwd)
echo ""
echo "Start running ==== python embed_dkn.py ===="

python3 embed_dkn.py \
--learning_rate 0.0001 \
--loss_weight 1.0 \
--max_click_history 8 \
--num_epochs 1 \
--use_entity True \
--use_context 0 \
--max_title_length 16 \
--entity_dim 128 \
--word_dim 300 \
--batch_size 128 \
--perform_shuffle 1 \
--data_dir ./model-update-dkn \
--checkpointPath ./model-update-dkn/temp/  \
--servable_model_dir ./model-update-dkn/model_complete/

if [[ $? -ne 0 ]]; then
  echo "error!!!"
  exit 1
fi

echo "run 'python embed_dkn.py' successfully"

ls ./model-update-dkn/model_complete/*/*

# ./model-update-dkn/model_complete/temp-1618990972/saved_model.pb
model_file=$(ls ./model-update-dkn/model_complete/*/saved_model.pb)
echo $model_file
model_dir_name=$(dirname ${model_file})
mkdir -p ./model-update-dkn/model_latest
mv ${model_dir_name}/* ./model-update-dkn/model_latest/
cd ./model-update-dkn/model_latest
echo "files in ./model-update-dkn/model_latest/"
ls -l
tar -cvf ../model.tar *
if [[ $? -ne 0 ]]; then
  echo "error!!!"
  exit 1
fi
mv ../model.tar .
gzip model.tar

if [[ $? -ne 0 ]]; then
  echo "error!!!"
  exit 1
fi

echo "Done ==== python embed_dkn.py ===="

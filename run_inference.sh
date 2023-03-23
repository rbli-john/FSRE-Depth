weight_folder=$1
config=$2
python evaluate_depth.py --load_weights_folder ${weight_folder} --config ${config} --data_path predictions/ --eval_split benchmark

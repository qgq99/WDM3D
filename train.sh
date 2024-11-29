export CUDA_VISIBLE_DEVICES=1

cur_time=$(date "+%Y-%m-%d_%H_%M_%S")
config_file=/home/qinguoqing/project/WDM3D/config/exp/exp.yaml
batch_size=6
epoch=1
output_dir=./output/train/${cur_time}_${epoch}e_${batch_size}bs

python script/train.py \
    --config ${config_file} \
    --batch_size ${batch_size} \
    --epoch ${epoch}
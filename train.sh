export CUDA_VISIBLE_DEVICES=0


title=GENeck_validdation
desc=进程已结束，退出代码为


cur_time=$(date "+%Y-%m-%d_%H_%M_%S")
config_file=/home/qinguoqing/project/WDM3D/config/exp/exp.yaml
batch_size=4
epoch=1
output_dir=/home/qinguoqing/project/WDM3D/output/train/${title}_${cur_time}

python script/train.py \
    --title ${title} \
    --desc "${desc}" \
    --config ${config_file} \
    --batch_size ${batch_size} \
    --epoch ${epoch} \
    --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
    --output_dir ${output_dir}
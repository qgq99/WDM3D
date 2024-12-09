export CUDA_VISIBLE_DEVICES=1


title=check_depth_mode_pred
desc="observe loss curve of a hole training process, with using depth mode: pred, which means the depth label is generated by a depther, the benefit is that the pred depth does not conain pixels without deppth."


cur_time=$(date "+%Y-%m-%d_%H_%M_%S")
config_file=/home/qinguoqing/project/WDM3D/config/exp/exp.yaml
batch_size=4
epoch=65
output_dir=/home/qinguoqing/project/WDM3D/output/train/${title}_${cur_time}

nohup python script/train.py \
    --title ${title} \
    --desc "${desc}" \
    --config ${config_file} \
    --batch_size ${batch_size} \
    --epoch ${epoch} \
    --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
    --output_dir ${output_dir} > console.txt &

# 调试
# python script/train.py \
#     --title ${title} \
#     --desc "${desc}" \
#     --config ${config_file} \
#     --batch_size ${batch_size} \
#     --epoch ${epoch} \
#     --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
#     --output_dir ${output_dir}
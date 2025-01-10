export CUDA_VISIBLE_DEVICES=1


title="regular_train"
desc="调整了伪点云生成方法, 将所有y坐标*-1"


cur_time=$(date "+%Y-%m-%d_%H_%M_%S")
# config_file=/home/qinguoqing/project/WDM3D/config/exp/exp.yaml
config_file=/home/qinguoqing/project/WDM3D/config/exp/exp_depth_off.yaml
batch_size=4
epoch=50
output_dir=/home/qinguoqing/project/WDM3D/output/train/${title}_${cur_time}

# nohup python script/train.py \
#     --title "${title}" \
#     --desc "${desc}" \
#     --config ${config_file} \
#     --batch_size ${batch_size} \
#     --epoch ${epoch} \
#     --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
#     --output_dir ${output_dir} > console.txt &

# 调试
# python script/train.py \
#     --title ${title} \
#     --desc "${desc}" \
#     --config ${config_file} \
#     --batch_size ${batch_size} \
#     --epoch ${epoch} \
#     --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
#     --output_dir ${output_dir}



nohup python script/train_depth_off.py \
    --title "${title}" \
    --desc "${desc}" \
    --config ${config_file} \
    --batch_size ${batch_size} \
    --epoch ${epoch} \
    --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
    --output_dir ${output_dir} > console.txt &


# 调试depth_off
# python script/train_depth_off.py \
#     --title ${title} \
#     --desc "${desc}" \
#     --config ${config_file} \
#     --batch_size ${batch_size} \
#     --epoch ${epoch} \
#     --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
#     --output_dir ${output_dir}
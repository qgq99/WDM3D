export CUDA_VISIBLE_DEVICES=1


title="train_with_backbone_ckpt"
desc="同时使用backbone的2d detector的ckpt进行训练，得到的权重用于查看可视化效果"


cur_time=$(date "+%Y-%m-%d_%H_%M_%S")
config_file=/home/qinguoqing/project/WDM3D/config/exp/exp.yaml
batch_size=4
epoch=15
output_dir=/home/qinguoqing/project/WDM3D/output/train/${title}_${cur_time}

nohup python script/train.py \
    --title "${title}" \
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
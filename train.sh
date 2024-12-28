export CUDA_VISIBLE_DEVICES=1


title="regular_train"
desc="将depth head提前到neck之前, 使得slope map在forward过程中计算"


cur_time=$(date "+%Y-%m-%d_%H_%M_%S")
config_file=/home/qinguoqing/project/WDM3D/config/exp/exp.yaml
batch_size=4
epoch=30
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
python script/train.py \
    --title ${title} \
    --desc "${desc}" \
    --config ${config_file} \
    --batch_size ${batch_size} \
    --epoch ${epoch} \
    --CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
    --output_dir ${output_dir}
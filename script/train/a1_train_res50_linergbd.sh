# Fail the script if there is any failure
set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi

# The name of this experiment.
name=$1

# Save logs and models under snap/gqa; make backup.
#resume_from='/home/liangyuan04/workspace/git/segmentation/line_segm/letr-depth-center/exp/lines_center_depth_1/checkpoints/checkpoint0499.pth'
# resume_from='/mnt/lab/liangyuan04/train_log/letr-depth-center/exp/glassline_1175/checkpoints/checkpoint0499.pth'
# output="/mnt/lab/liangyuan04/train_log/letr-depth-center/exp/$name"
#resume_from="exp/glassline_1175/checkpoints/checkpoint0499.pth"

resume_from='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
#resume_from='/mnt/lab/data/liangyuan/train_log/glass-depth/exp/samplenumtest3-oddinterval3080-60_160/checkpoints/checkpoint0164.pth'
output="/mnt/lab/data/liangyuan/train_log/glass-depth/exp/$name"
if [ ! -d "$output"  ]; then
    echo "folder not exist"
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    PYTHONPATH=$PYTHONPATH:./src python -m torch.distributed.launch \
    --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=2 --use_env  src/main_glassrgbd.py \
    --output_dir $output --backbone resnet50 --resume $resume_from \
    --batch_size 1 ${@:2} --epochs 200 --lr_drop 70 --num_queries 100 --num_gpus 4 \
    --with_line --with_center --with_dense \
    --log_depth_error | tee -a $output/history.txt  

else
    echo "$output already exist"
fi




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
resume_from="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth" 
#resume_from='/mnt/lab/liangyuan04/train_log/letr-depth-center/exp/line_baseline_denseEdge_onegpu/checkpoints/checkpoint0435.pth'
output="/mnt/lab/liangyuan04/train_log/letr-depth-center/exp/$name"
if [ ! -d "$output"  ]; then
    echo "folder not exist"
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    PYTHONPATH=$PYTHONPATH:./src python -m torch.distributed.launch \
    --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=4 --use_env  src/main_glassrgbd.py \
    --output_dir $output --backbone resnet50 --resume $resume_from --no_opt \
    --batch_size 1 ${@:2} --epochs 150 --lr_drop 50 --num_queries 100 --num_gpus 4 \
    --with_dense | tee -a $output/history.txt  

else
    echo "folder already exist"
fi




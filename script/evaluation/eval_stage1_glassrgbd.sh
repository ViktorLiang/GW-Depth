# Fail the script if there is any failure
set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
name=$1
output=exp/$name

epoch=("499")

for ((i=0;i<${#epoch[@]};++i)); do
    mkdir  -p $output/score
    mkdir  -p $output/score/eval-sAP
    mkdir  -p $output/score/eval-fscore
    mkdir  -p $output/score/eval-sAP-york
    mkdir  -p $output/score/eval-fscore-york

    CUDA_VISIBLE_DEVICES=0
    PYTHONPATH=$PYTHONPATH:./src \
    python ./src/main_glassrgbd.py \
    --output_dir $output --backbone resnet50 --resume $output/checkpoints/checkpoint0${epoch[i]}.pth \
    --batch_size 1 ${@:2}  --num_queries 100 \
    --eval --benchmark --dataset val --append_word ${epoch[i]} --no_aux_loss \
    --dataset_args_file ./script/train/arguments_train_glassrgbd.txt \
    --min_depth_eval 1e-3 \
    --max_depth_eval 10


    # python evaluation/eval-sAP-glassrgbd.py $output/benchmark/benchmark_val_${epoch[i]} | tee $output/score/eval-sAP/${epoch[i]}.txt

    # python evaluation/eval-fscore-glassrgbd.py $output/benchmark/benchmark_val_${epoch[i]} | tee $output/score/eval-fscore/${epoch[i]}.txt


done


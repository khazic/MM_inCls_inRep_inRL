#!/bin/bash
export WANDB_API_KEY=9c69c18b00c7dac67189f39e261a257ebd476cda
source activate /opt/conda/envs/verl
cd /liuchonghan/Multimodal_Search/example

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)

torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID ../src/run_vl_classification.py \
    --model_name_or_path /SHARE/liuchonghan/Qwen2-VL-2B-Instruct \
    --train_file ../data/cls_data/train_dataset.json \
    --validation_file ../data/cls_data/train_dataset.json \
    --label_file ../data/cls_data/label.json \
    --output_dir ../experiment/Qwen2VL-cls/ \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_eval_samples 2500 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_steps 4000 \
    --logging_steps 3 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --dataloader_num_workers 16 \
    --remove_unused_columns False \
    --lr_scheduler_type cosine \
    --adam_epsilon 1e-6 \
    --optim adamw_torch_fused \
    --warmup_steps 200 \
    --report_to wandb \
    # --load_best_model_at_end \
    "$@"
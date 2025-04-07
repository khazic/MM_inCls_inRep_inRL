export WANDB_API_KEY=9c69c18b00c7dac67189f39e261a257ebd476cda
source activate /opt/conda/envs/verl

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)

torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID \
    ../../src/tasks/classification/trainers/run_qwen2vl.py \
    --model_name_or_path /SHARE/liuchonghan/Qwen2-VL-2B-Instruct \
    --train_file ../../data/cls_data/train_dataset.json \
    --validation_file ../../data/cls_data/train_dataset.json \
    --label_file ../../data/cls_data/label.json \
    --output_dir ../../experiment/Qwen2VL-cls/ \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --max_eval_samples 2500 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_steps 5000 \
    --logging_steps 3 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --dataloader_num_workers 24 \
    --remove_unused_columns False \
    --lr_scheduler_type cosine \
    --adam_epsilon 1e-6 \
    --optim adamw_torch_fused \
    --warmup_steps 200 \
    --report_to wandb \
    --model_type qwen2vl \
    --problem_type single_label_classification   # "regression" "multi_label_classification" "single_label_classification"
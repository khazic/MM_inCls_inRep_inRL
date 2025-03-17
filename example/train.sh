export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
<<<<<<< HEAD

export WANDB_DIR=/SHARE/liuchonghan/Multimodal_Search/wandb

torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID ../src/run_vl_classification.py \
    --model_name_or_path /SHARE/liuchonghan/Qwen2-VL-2B-Instruct \
    --train_file ../data/cls_data/train_dataset.json \
    --validation_file ../data/cls_data/train_dataset.json \
    --label_file ../data/cls_data/label.json \
=======
export OMP_NUM_THREADS=1
torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID  run_vl_classification.py \
    --model_name_or_path /SHARE/liuchonghan/Qwen2.5vl_3b \
    --train_file ./data/train_dataset.json \
    --validation_file ./data/train_dataset.json \
>>>>>>> dc69ea49ec8bd822cbdbf2600dd844f20e8b75db
    --output_dir ../experiment/QwenVL-cls/ \
    --learning_rate 5e-6 \
    --num_train_epochs 10 \
    --max_eval_samples 2000 \
    --per_device_train_batch_size 18 \
    --per_device_eval_batch_size 18 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 2000 \
    --logging_steps 5 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --dataloader_num_workers 16 \
    --remove_unused_columns False \
    --lr_scheduler_type cosine \
    --adam_epsilon 1e-6 \
    --optim adamw_torch_fused \
    --warmup_steps 5000 \
<<<<<<< HEAD
    --report_to wandb \
    # --load_best_model_at_end \
    "$@"
=======
    --label_file ./data/label.json \
    --report_to wandb ../experiment/QwenVL-cls/wandb_logs \
    "$@"

#     --load_best_model_at_end \
>>>>>>> dc69ea49ec8bd822cbdbf2600dd844f20e8b75db

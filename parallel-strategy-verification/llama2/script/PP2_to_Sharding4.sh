#!/bin/bash

#跑shardingV2的时候用：--sharding_parallel_config "split_param" \

# 设置环境变量
export PYTHONPATH=../:$PYTHONPATH
export FLAGS_call_stack_level=3
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_cudnn_deterministic=True
export FLAGS_embedding_deterministic=1 


ROOT_DIR="/home/aistudio/PaddleNLP/llm"

# 转换前训练
task_name="PP2_to_Sharding4"
# 从task_name中提取TP2和TP4作为曲线名称
curve_name1=$(echo $task_name | cut -d'_' -f1)  # 提取TP2
curve_name2=$(echo $task_name | cut -d'_' -f3)  # 提取TP4

case_temp0_out_dir="${ROOT_DIR}/temp0/${task_name}"
case_temp0_log_dir="${ROOT_DIR}/temp0/${task_name}_log"

# 清理旧的输出目录
rm -rf $case_temp0_out_dir
rm -rf $case_temp0_log_dir

# train 
python -u -m paddle.distributed.launch \
    --gpus "0,1" \
    --log_dir "$case_temp0_log_dir" \
    run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-7b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-7b" \
    --input_dir "./data" \
    --split "949,50,1" \
    --num_hidden_layers 4 \
    --output_dir "$case_temp0_out_dir" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 2 \
    --tensor_parallel_config "enable_delay_scale_loss enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv enable_overlap_p2p_comm" \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --enable_linear_fused_grad_add 0 \
    --learning_rate 3e-05 \
    --logging_steps 1 \
    --max_steps 51 \
    --save_steps 50 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --fp16 1 \
    --fp16_opt_level "O2" \
    --amp_master_grad 1 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute 0 \
    --save_total_limit 2 \
    --device "gpu" \
    --save_sharded_model 0 \
    --using_flex_checkpoint 1 \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --unified_checkpoint 0 \
    # --sharding_parallel_degree 2 \
    # --sharding "stage1" \
    # --sharding_parallel_config "split_param" \


export FLAGS_shard_bypass_dygraph_optimizer=1


# 从转换前训练load
case_temp1_out_dir="${ROOT_DIR}/temp1/${task_name}"
case_temp1_log_dir="${ROOT_DIR}/temp1/${task_name}_log"

# 清理旧的输出目录
rm -rf $case_temp1_out_dir
rm -rf $case_temp1_log_dir


python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "$case_temp1_log_dir" \
    run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-7b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-7b" \
    --input_dir "./data" \
    --split "949,50,1" \
    --num_hidden_layers 4 \
    --output_dir "$case_temp1_out_dir" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --tensor_parallel_config "enable_delay_scale_loss enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv enable_overlap_p2p_comm" \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --enable_linear_fused_grad_add 0 \
    --learning_rate 3e-05 \
    --logging_steps 1 \
    --max_steps 51 \
    --save_steps 51 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --fp16 1 \
    --fp16_opt_level "O2" \
    --amp_master_grad 1 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute 0 \
    --save_total_limit 2 \
    --device "gpu" \
    --save_sharded_model 0 \
    --using_flex_checkpoint 1 \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --unified_checkpoint 0 \
    --resume_from_checkpoint "${case_temp0_out_dir}/checkpoint-50" \
    --sharding_parallel_degree 4 \
    --sharding "stage1" \
    # --sharding_parallel_config "split_param" \


# load回
case_temp2_out_dir="${ROOT_DIR}/temp2/${task_name}"
case_temp2_log_dir="${ROOT_DIR}/temp2/${task_name}_log"

# 清理旧的输出目录
rm -rf $case_temp2_out_dir
rm -rf $case_temp2_log_dir

python -u -m paddle.distributed.launch \
    --gpus "0,1" \
    --log_dir "$case_temp2_log_dir" \
    run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-7b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-7b" \
    --input_dir "./data" \
    --split "949,50,1" \
    --num_hidden_layers 4 \
    --output_dir "$case_temp2_out_dir" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 2 \
    --tensor_parallel_config "enable_delay_scale_loss enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv enable_overlap_p2p_comm" \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --enable_linear_fused_grad_add 0 \
    --learning_rate 3e-05 \
    --logging_steps 1 \
    --max_steps 52 \
    --save_steps 52 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --fp16 1 \
    --fp16_opt_level "O2" \
    --amp_master_grad 1 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute 0 \
    --save_total_limit 2 \
    --device "gpu" \
    --save_sharded_model 0 \
    --using_flex_checkpoint 1 \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --unified_checkpoint 0 \
    --resume_from_checkpoint "${case_temp1_out_dir}/checkpoint-51"
    # --sharding_parallel_degree 2 \
    # --sharding "stage1" \
    # --sharding_parallel_config "split_param" \

#比较转换前，后转换回来的ckpt的md5是否完全一致，若完全一致，则会输出：MD5匹配通过
python -m compare_checkpoints temp2/${task_name}/checkpoint-52 temp0/${task_name}/checkpoint-50




# load to train

export FLAGS_shard_bypass_dygraph_optimizer=0

case_temp3_out_dir="${ROOT_DIR}/temp3/${task_name}"
case_temp3_log_dir="${ROOT_DIR}/temp3/${task_name}_log"

# 清理旧的输出目录
rm -rf $case_temp3_out_dir
rm -rf $case_temp3_log_dir

python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "$case_temp3_log_dir" \
    run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-7b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-7b" \
    --input_dir "./data" \
    --split "949,50,1" \
    --num_hidden_layers 4 \
    --output_dir "$case_temp3_out_dir" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --tensor_parallel_config "enable_delay_scale_loss enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv enable_overlap_p2p_comm" \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --enable_linear_fused_grad_add 0 \
    --learning_rate 3e-05 \
    --logging_steps 1 \
    --max_steps 200 \
    --save_steps 201 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --fp16 1 \
    --fp16_opt_level "O2" \
    --amp_master_grad 1 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute 0 \
    --save_total_limit 2 \
    --device "gpu" \
    --save_sharded_model 0 \
    --using_flex_checkpoint 1 \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --unified_checkpoint 0 \
    --resume_from_checkpoint "${case_temp0_out_dir}/checkpoint-50" \
    --sharding_parallel_degree 4 \
    --sharding "stage1" \
    # --sharding_parallel_config "split_param" \

# 计算续训的 loss diff 精度误差
python -m coculate_loss_with_md5 ${case_temp0_log_dir}/workerlog.0 ${case_temp3_log_dir}/workerlog.0 
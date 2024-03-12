##Evaluating Reconstruction
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port=15200 eval_reconstruction.py \
    --batch_size 16 \
    --image_size 128 \
    --n_class 1000 \
    --imagenet_path $imagenet_path \
    --n_vision_words 32000 \
    --vq_config_path vqgan_configs/v2l.yaml \
    --output_dir $log_dir \
    --log_dir $log_dir \
    --quantizer_type "org" \
    --local_embedding_path "codebooks/local_codebook_embedding.pth" \
    --global_embedding_path "codebooks/global_codebook_embedding.pth" \
    --stage_1_ckpt "checkpoints/v2l-decode.pth" \
    --embed_dim 768 \
    --use_cblinear 1 \
    --use_crossatt_dec 1

##Evaluating Few-shot Classification (N-way K-Shot M-Repeat)
token_nums=(5 21)
ways=(5 2)
shots=(3 5 1)
for (( k = 0 ; k < ${#token_nums[@]} ; k++ ))
do
    token_num=${token_nums[$k]}
    for (( i = 0 ; i < ${#ways[@]} ; i++ ))
    do
        way=${ways[$i]}
        for (( j = 0 ; j < ${#shots[@]} ; j++ ))
        do
        shot=${shots[$j]}
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port=10645 eval_understanding.py \
                    --image_size 128 \
                    --n_class 1000 \
                    --batch_size 1 \
                    --max_seq_len 1024 \
                    --num_workers 0 \
                    --output_type "next_token_prediction" \
                    --mini_imagenet_path "/data/all" \
                    --vq_config_path vqgan_configs/v2l.yaml \
                    --output_dir "log_eval_few_shot/7B_"$token_num"_"$way"_"$shot \
                    --llama_model_path /data/llama2-origin-format/llama-2-7b \
                    --induction 1 \
                    --stage_1_ckpt "checkpoints/v2l-decode.pth" \
                    --embed_dim 768 \
                    --quantizer_type "org" \
                    --use_cblinear 1 \
                    --use_crossatt_dec 1 \
                    --local_embedding_path "codebooks/local_codebook_embedding.pth" \
                    --global_embedding_path "codebooks/global_codebook_embedding.pth" \
                    --way $way \
                    --shot $shot \
                    --token_num $token_num \
                    --repeat 0
        done
    done
done

##Evaluating Denoising Generation
tasks=("deblur" "rotation" "shift" "inpainting" "outpainting")
for (( k = 0 ; k < ${#tasks[@]} ; k++ ))
do
    task=${tasks[$k]}
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port=12593 eval_denoising_generation.py \
                            --image_size 128 \
                            --n_class 1000 \
                            --max_seq_len 512 \
                            --num_workers 0 \
                            --output_type "next_token_prediction" \
                            --vq_config_path vqgan_configs/v2l.yaml \
                            --output_dir "log_eval_"$task"/7B_clip_linear" \
                            --quantizer_type "org" \
                            --llama_model_path /data/llama2-origin-format/llama-2-7b \
                            --embed_dim 768 \
                            --n_vision_words 32000 \
                            --local_embedding_path "codebooks/local_codebook_embedding.pth" \
                            --global_embedding_path "codebooks/global_codebook_embedding.pth" \
                            --stage_1_ckpt "checkpoints/v2l-decode.pth" \
                            --batch_size 1 \
                            --global_token_num 21 \
                            --prompt_length 16 \
                            --step 2 \
                            --use_cblinear 1 \
                            --use_crossatt_dec 1 \
                            --task $task
done


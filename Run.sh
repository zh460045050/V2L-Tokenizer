imagenet_path=""
log_dir=""
llama_path=""

####Expand Global Vocabulary Set
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=12247 step1_epanding_vocabulary_set.py \
    --batch_size 400 \
    --max_seq_len 64 \
    --llama_model_path $llama_path

####Generate Embedding for Vocabularies
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=12247 step2_generate_codebook_embedding.py

####Filtering Vocabularies with Training Images
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=12247 step3_global_codebook_filtering.py \
    --batch_size 2 \
    --imagenet_path $imagenet_path \
    --num_workers 16

####Training V2L Tokenizer
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port=12247 step4_training_v2l_tokenizer.py \
    --batch_size 8 \
    --image_size 128 \
    --epochs 100 \
    --warmup_epochs 5 \
    --lr 4.5e-4 \
    --n_class 1000 \
    --imagenet_path $imagenet_path \
    --num_workers 16 \
    --rate_q 1 \
    --rate_p 0.1 \
    --vq_config_path vqgan_configs/v2l.yaml \
    --output_dir $log_dir \
    --log_dir $log_dir \
    --disc_start 10000 \
    --local_embedding_path codebooks/local_codebook_embedding.pth \
    --global_embedding_path codebooks/global_codebook_embedding.pth \
    --tuning_codebook 0 \
    --use_cblinear 1 \
    --use_crossatt_dec 1 \
    --embed_dim 768


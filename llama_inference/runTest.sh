torchrun --nproc_per_node 1 example_text_completion_debug.py \
    --ckpt_dir /data/llama2-origin-format/llama-2-7b/ \
    --tokenizer_path /data/llama2-origin-format/llama-2-7b/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
cd path_to_project/
python -m torch.distributed.run --nproc_per_node 8 --master_port 2024 main_pretrain.py \
    --batch_size 64 --input_size 512 --token_size 32 \
    --model mae_swin_large_512 --mask_regular \
    --epochs 256 --warmup_epochs 10 --blr 1e-4 \
    --data_path path_to_dataset/ \
    --log_dir path_to_log/ \
    --output_dir path_to_output/ \
    --load_from path_to_checkpoint


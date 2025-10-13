EPOCHS=50
for LABLE in HDLC
do
for BLR in 5e-3
do
cd path_to_project_folder
python -m torch.distributed.run --nproc_per_node 1 --master_port 2024 main_finetune.py \
		--input_size 256 \
		--batch_size 32 \
		--model swin_large_512to256 \
		--finetune path_to_checkpoint_file \
		--epochs ${EPOCHS} \
		--early_stop 12 \
		--blr ${BLR} \
        --dataset xxx_single \
        --data_path path_to_data_folder/ \
        --annotation_path path_to_annotation_file \
        --label ${LABLE} \
		--type regression \
		--nb_classes 1 \
		--output_dir path_to_output_folder/
done
done
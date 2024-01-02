export CUDA_VISIBLE_DEVICES=0,1,3

torchrun --nnodes 1 --nproc_per_node 1 examples/finetuning.py \ #--enable_fsdp \
	--model_name /data/liuda/tmp/llama-recipes/models/Llama-2-7b-hf \
	--use_peft --peft_method lora \
	--use_fp16 \
	--output_dir /data/liuda/tmp/llama-recipes/models/Lora \
	--dataset alpaca_dataset --data_path /data/liuda/tmp/llama-recipes/ft_datasets/alpaca_data.json \
	--quantization \
	--batch_size_training 10 --num_epochs 1

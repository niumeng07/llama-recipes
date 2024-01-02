export CUDA_VISIBLE_DEVICES=0

python3.10 -m llama_recipes.finetuning \
	--use_peft --peft_method lora \
	--quantization \
	--model_name /data/liuda/tmp/llama-recipes/models/Llama-2-7b-hf \
	--output_dir /data/liuda/tmp/llama-recipes/models/Lora \
	--dataset alpaca_dataset \
	--data_path /data/liuda/tmp/llama-recipes/ft_datasets/alpaca_data.json \
	--batch_size_training 10 \
	--num_epochs 1 
	# --nnodes 1 --nproc_per_node 3 --enable_fsdp --fsdp_config.pure_bf16


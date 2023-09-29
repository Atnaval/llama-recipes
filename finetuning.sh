python -m llama_recipes.finetuning \
--use_peft --peft_method lora --quantization \
--model_name meta-llama/Llama-2-13b-hf --output_dir ./output/13b \
--dataset lawsum_dataset \
--enable_wandb --wandb_project sum-law --wandb_run 2_13b \
--batch_size_training 64
--val_batch_size 16
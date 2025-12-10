MODEL_NAME="Qwen/Qwen2.5-0.5B"
# MODEL_NAME="mistralai/Mistral-7B-v0.1"

CUDA_VISIBLE_DEVICES=5 \
accelerate launch \
  --config_file accelerate_config.yaml \
  -m embedding.main \
  --model_name_or_path=$MODEL_NAME \
  --per_device_train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --max_new_tokens=5 \
  --logit_temp=1 \
  --wandb_project="MY-Project" \
  --model_type="generation" \
  --pooling_method="generate_mean" \
  --save_steps=1000 \
  --data_sampling_rate=0.05 \
  --reg_weight=1 \
  --output_dir ckpts/GIRCSE-QWEN0.5B

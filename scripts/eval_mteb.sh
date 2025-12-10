MODEL_NAME="Qwen/Qwen2.5-0.5B"
# MODEL_NAME="mistralai/Mistral-7B-v0.1"

# Define checkpoint base path
CHECKPOINT="your/checkpoint/directory/path"

CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
    --config_file accelerate_config.yaml \
    -m embedding.mteb_eval \
    --model_name $MODEL_NAME \
    --checkpoint_path "$CHECKPOINT" \
    --max_new_tokens 3 \
    --batch_size 64 \
    --pooling_method generate_mean \
    --trainer_type soft

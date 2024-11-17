export MODEL_NAME="timbrooks/instruct-pix2pix" 
# timbrooks/instruct-pix2pix
export OUTPUT_DIR="./output/modelname"
export TRAIN_DIR="/dataset/"  #--train_data_dir=$TRAIN_DIR \
# export DATASET_ID="fusing/instructpix2pix-1000-samples"   #--dataset_name=$DATASET_ID \      # --rank=64 \
# --use_ema \

accelerate launch --main_process_port=29501 --mixed_precision="fp16"  train_instruct_pix2pix_lora.py \
    --use_expand_prompt=False --plip_or_clip=None \
    --use_adapter=False \
    --json_file='metadata-thinlung-HE.jsonl' \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --rank=128 \
    --train_data_dir=$TRAIN_DIR \
    --output_dir=${OUTPUT_DIR} \
    --resolution=512 \
    --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=100000 \
    --checkpointing_steps=10000 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --learning_rate=1e-04  --lr_warmup_steps=0 \
    --mixed_precision="fp16" 
    


    # --resume_from_checkpoint="latest"
    
    # --push_to_hub

# accelerate launch train_instruct_pix2pix_sdxl.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --dataset_name=$DATASET_ID \
#     --enable_xformers_memory_efficient_attention \
#     --resolution=256 --random_flip \
#     --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
#     --max_train_steps=15000 \
#     --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#     --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
#     --conditioning_dropout_prob=0.05 \
#     --seed=42 \
#     --push_to_hub
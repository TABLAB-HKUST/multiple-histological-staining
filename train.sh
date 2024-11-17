export MODEL_NAME="timbrooks/instruct-pix2pix" 
export OUTPUT_DIR="./output/modelname"
export TRAIN_DIR="/dataset/"  #--train_data_dir=$TRAIN_DIR \

accelerate launch  --mixed_precision="fp16"  train_instruct_lora.py \
    --json_file='metadata.jsonl' \
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

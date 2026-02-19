#!/bin/bash

# --- CONFIGURATION ---
dataset="mimic"
model_type="meta-llama/Llama-3.2-3B-Instruct"
feat_ext="stsb-roberta-base-v2" # Must match Step 2
api="HFGPT"

# Privacy & Training Params
noise=2.36          # <--- REPLACE WITH VALUE FROM STEP 3
epochs=10            # Number of PE iterations
num_seed_samples=2000 # Samples per epoch
mlm_prob=0.5        # Variation degree (keep 0.5 or 0 for no variation)
L=7
init_L=7
lookahead_degree=0
select_syn_mode="rank"

# Calc total samples
num_samples=$((L*num_seed_samples))

# Paths
# Note: Ensure this matches where Step 2 saved the file
embedding_file="result/embeddings/${feat_ext}/${dataset}_train_all.embeddings.npz"
train_data_file="data/${dataset}/train.csv"

# Result folder name
result_folder="result/${dataset}/llama3_${feat_ext}/eps${noise}_epochs${epochs}"

echo "------------------------------------------------"
echo "Running Aug-PE on ${dataset}"
echo "Model: ${model_type}"
echo "Embeddings: ${embedding_file}"
echo "Output: ${result_folder}"
echo "Lookahead: ${lookahead_degree}"
echo "------------------------------------------------"

# --- MAIN LOOP ---
# This loop checks if previous iterations exist to resume training
data_checkpoint_args=""
for (( iter=0; iter<=epochs; iter++ ))
do
    train_file=${result_folder}/${iter}/samples.csv
    if [ -e "$train_file" ]; then
        echo "Found checkpoint: $train_file"
        data_checkpoint_args="--data_checkpoint_step ${iter} --data_checkpoint_path ${train_file}"
    fi
done

# --- RUN MAIN.PY ---
# Note: We pass --model_type for Llama
# We pass --dataset mimic so data_loader uses your custom logic
python main.py ${data_checkpoint_args} \
    --api ${api} \
    --dataset ${dataset} \
    --train_data_file ${train_data_file} \
    --train_data_embeddings_file ${embedding_file} \
    --model_type ${model_type} \
    --num_nearest_neighbor 5 \
    --noise_multiplier ${noise} \
    --epochs ${epochs} \
    --num_samples_schedule ${num_samples} \
    --variation_degree_schedule ${mlm_prob} \
    --combine_divide_L ${L} \
    --init_combine_divide_L ${init_L} \
    --lookahead_degree ${lookahead_degree} \
    --lookahead_self \
    --select_syn_mode ${select_syn_mode} \
    --nn_mode cos_sim \
    --feature_extractor ${feat_ext} \
    --feature_extractor_batch_size 1024 \
    --result_folder ${result_folder} \
    --save_syn_mode all \
    --fp16 \
    --do_sample

echo "Generation Complete."

# --- EVALUATION ---
last_iter=${epochs}
final_synthetic_data="${result_folder}/${last_iter}_all/samples.csv"

echo "================================================"
echo "Running Evaluation on: ${final_synthetic_data}"
echo "  - Train on synthetic, test on real"
echo "  - Train on real (baseline upper bound)"
echo "  - Privacy metrics (leakage, memorization)"
echo "================================================"

python evaluate_mimic.py \
    --synthetic-data ${final_synthetic_data} \
    --output-dir ${result_folder}/eval_results \
    --real-baseline

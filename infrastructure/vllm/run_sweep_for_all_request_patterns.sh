#!/bin/bash
set -e

CONFIG="configs/gpu_model_map_h100.yaml"
OUTPUT="data/h100_full_sweep.csv"
NUM_PROMPTS=8000
INTERVAL=0.2
CONCURRENCY=120

mkdir -p logs

#############################
# 1️⃣ Poisson – low/med/high
#############################

# Low load
RATE=4
LOGFILE="logs/run_poisson_low_${RATE}.log"
echo ">>> Running Poisson Low, rate=${RATE}"
python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern poisson \
  --rate ${RATE} \
  --concurrency ${CONCURRENCY} \
  > ${LOGFILE} 2>&1

# Medium load
RATE=10
LOGFILE="logs/run_poisson_med_${RATE}.log"
echo ">>> Running Poisson Medium, rate=${RATE}"
python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern poisson \
  --rate ${RATE} \
  --concurrency ${CONCURRENCY} \
  > ${LOGFILE} 2>&1

# High load
RATE=18
LOGFILE="logs/run_poisson_high_${RATE}.log"
echo ">>> Running Poisson High, rate=${RATE}"
python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern poisson \
  --rate ${RATE} \
  --concurrency ${CONCURRENCY} \
  > ${LOGFILE} 2>&1


##########################################
# 2️⃣ Microburst – short extreme overload
##########################################

LOGFILE="logs/run_microburst.log"
echo ">>> Running microburst (base=6)"
python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern microburst \
  --rate 6 \
  --concurrency ${CONCURRENCY} \
  > ${LOGFILE} 2>&1


##########################################
# 3️⃣ Sustained overload – heavy pressure
##########################################

LOGFILE="logs/run_sustained.log"
echo ">>> Running sustained overload, rate=25"
python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern sustained \
  --rate 25 \
  --concurrency ${CONCURRENCY} \
  > ${LOGFILE} 2>&1


echo "🎯 Sweep COMPLETE. Clean dataset at: ${OUTPUT}"

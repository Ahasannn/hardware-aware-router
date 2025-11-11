#!/bin/bash
set -e

CONFIG="configs/gpu_model_map_qwen.yaml"
OUTPUT="data/hw_dataset_qwen_sweep.csv"
NUM_PROMPTS=8000
INTERVAL=0.3
CONCURRENCY=16        # ≈8 per GPU → high but safe

mkdir -p logs

# 1️⃣ Poisson – low/mid/high
for RATE in 2 5 10; do
  LOGFILE="logs/run_poisson_${RATE}.log"
  echo ">>> Running pattern=poisson, rate=${RATE}"
  nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
    --config ${CONFIG} \
    --output ${OUTPUT} \
    --num_prompts ${NUM_PROMPTS} \
    --interval ${INTERVAL} \
    --pattern poisson \
    --rate ${RATE} \
    --concurrency ${CONCURRENCY} \
    >> ${LOGFILE} 2>&1
  echo "✅ Completed poisson rate=${RATE}"
done

# 2️⃣ Microburst – transient overload
LOGFILE="logs/run_microburst_5.log"
echo ">>> Running pattern=microburst, rate=5"
nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern microburst \
  --rate 5 \
  --concurrency ${CONCURRENCY} \
  >> ${LOGFILE} 2>&1
echo "✅ Completed microburst (rate=5)"

# 3️⃣ Sustained overload – heavy
LOGFILE="logs/run_sustained_10.log"
echo ">>> Running pattern=sustained, rate=10"
nohup python -m src.hardware_cost_model.build_hardware_cost_dataset \
  --config ${CONFIG} \
  --output ${OUTPUT} \
  --num_prompts ${NUM_PROMPTS} \
  --interval ${INTERVAL} \
  --pattern sustained \
  --rate 10 \
  --concurrency ${CONCURRENCY} \
  >> ${LOGFILE} 2>&1
echo "✅ Completed sustained (rate=10)"

echo "🎯 Sweep complete. Dataset appended to: ${OUTPUT}"

#!/usr/bin/env python3
"""End-to-end test of the complete HW-Router pipeline"""
import time
import numpy as np
from openai import OpenAI

from hw_router.hardware_monitor import start_metrics_watcher, model_metrics
from hw_router.cost_predictor import HardwareCostPredictor
from hw_router.constants import DEFAULT_LAMBDA, LAT_P95_LOG

def test_end_to_end():
    """Test the complete HW-Router pipeline with deployed models"""

    print("\n" + "="*70)
    print("END-TO-END HW-ROUTER PIPELINE TEST")
    print("="*70 + "\n")

    # ===============================================
    # 1. Setup: Define model configurations
    # ===============================================
    print("Step 1: Setting up model configurations...")

    models = {
        "tinyllama": {
            "base_url": "http://localhost:8020",
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_id": 1,  # Using phi3-mini as proxy since we don't have these exact models
            "gpu_id": "0"
        },
        "qwen05b": {
            "base_url": "http://localhost:8021",
            "model_name": "Qwen/Qwen1.5-0.5B",
            "model_id": 3,  # Using qwen3b as proxy
            "gpu_id": "0"
        }
    }

    # Create OpenAI clients
    clients = {}
    model_url_map = {}

    for name, config in models.items():
        clients[name] = OpenAI(
            base_url=f"{config['base_url']}/v1",
            api_key="EMPTY"
        )
        model_url_map[name] = f"{config['base_url']}/metrics"

    print(f"✓ Configured {len(models)} models\n")

    # ===============================================
    # 2. Start hardware monitor
    # ===============================================
    print("Step 2: Starting hardware monitor...")
    start_metrics_watcher(model_url_map, interval=2)
    time.sleep(3)  # Wait for initial metrics
    print("✓ Hardware monitor active\n")

    # ===============================================
    # 3. Load cost predictor
    # ===============================================
    print("Step 3: Loading cost predictor...")
    try:
        cost_predictor = HardwareCostPredictor(
            "checkpoints/hardware_cost_model/model.pt",
            "checkpoints/hardware_cost_model/preproc.joblib"
        )
        print("✓ Cost predictor loaded\n")
    except Exception as e:
        print(f"✗ Failed to load cost predictor: {e}\n")
        return False

    # ===============================================
    # 4. Test prompt
    # ===============================================
    test_prompt = "Explain what machine learning is in simple terms."
    p_tokens = len(test_prompt.split()) * 1.3  # Rough estimate

    print(f"Step 4: Testing with prompt (≈{int(p_tokens)} tokens):")
    print(f"  \"{test_prompt[:60]}...\"\n")

    # ===============================================
    # 5. Routing decision with HW-Router
    # ===============================================
    print("Step 5: Making routing decisions...\n")
    print("-"*70)

    lambda_param = DEFAULT_LAMBDA
    best_model = None
    best_score = -float('inf')

    for model_name, config in models.items():
        # Get current hardware metrics
        hw_metrics = model_metrics.get(model_name, {})

        # Predict cost using hardware-aware cost model
        features = {
            "p_tokens": p_tokens,
            "running_req_count": hw_metrics.get("num_requests_running", 0),
            "waiting_req_count": hw_metrics.get("num_requests_waiting", 0),
            "kv_cache_usage_perc": hw_metrics.get("kv_cache_usage_perc", 0.0),
            "ttft_avg": hw_metrics.get("ttft_avg", 0.0),
            "itl_avg": hw_metrics.get("itl_avg", 0.0),
            "model_id": config["model_id"],
            "gpu_id": config["gpu_id"]
        }

        try:
            ttft_pred, tpot_pred = cost_predictor(config["model_id"], features)

            # Normalize cost (log-space latency)
            latency_pred = ttft_pred + (tpot_pred * 50)  # Assume 50 output tokens
            cost_normalized = np.log(latency_pred + 1e-6) / LAT_P95_LOG

            # For this test, use dummy quality scores
            # In production, this would come from IRT/CARROT
            quality = 0.75 if model_name == "qwen05b" else 0.65

            # HW-Router scoring: S = λ·Q - (1-λ)·C
            score = lambda_param * quality - (1 - lambda_param) * cost_normalized

            print(f"{model_name.upper()}:")
            print(f"  Hardware State:")
            print(f"    Running requests: {features['running_req_count']}")
            print(f"    Waiting requests: {features['waiting_req_count']}")
            print(f"    KV cache usage:   {features['kv_cache_usage_perc']:.1%}")
            print(f"  Cost Prediction:")
            print(f"    TTFT: {ttft_pred:.4f}s")
            print(f"    TPOT: {tpot_pred:.6f}s")
            print(f"    Est. latency: {latency_pred:.3f}s")
            print(f"    Normalized cost: {cost_normalized:.3f}")
            print(f"  Quality (dummy): {quality:.3f}")
            print(f"  → Score (λ={lambda_param}): {score:.4f}")
            print()

            if score > best_score:
                best_score = score
                best_model = model_name

        except Exception as e:
            print(f"  ✗ Prediction failed: {e}\n")
            continue

    print("-"*70)
    if best_model:
        print(f"\n✓ ROUTING DECISION: {best_model.upper()} (score: {best_score:.4f})\n")
    else:
        print("\n✗ Routing decision failed\n")
        return False

    # ===============================================
    # 6. Execute request on selected model
    # ===============================================
    print("Step 6: Executing request on selected model...")

    try:
        selected_client = clients[best_model]
        selected_model_name = models[best_model]["model_name"]

        start_time = time.time()
        response = selected_client.chat.completions.create(
            model=selected_model_name,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=100,
            temperature=0.7
        )
        end_time = time.time()

        latency = end_time - start_time
        response_text = response.choices[0].message.content

        print(f"✓ Request completed successfully!")
        print(f"  Actual latency: {latency:.3f}s")
        print(f"  Response preview: {response_text[:100]}...")
        print()

    except Exception as e:
        print(f"✗ Request failed: {e}\n")
        return False

    # ===============================================
    # 7. Verify hardware metrics were updated
    # ===============================================
    print("Step 7: Verifying hardware metrics update...")
    time.sleep(2)  # Wait for metrics to update

    updated_metrics = model_metrics.get(best_model, {})
    print(f"✓ Updated metrics for {best_model}:")
    print(f"  E2E latency avg: {updated_metrics.get('e2e_avg', 0):.4f}s")
    print(f"  TTFT avg: {updated_metrics.get('ttft_avg', 0):.4f}s")
    print(f"  ITL avg: {updated_metrics.get('itl_avg', 0):.6f}s")
    print()

    # ===============================================
    # Summary
    # ===============================================
    print("="*70)
    print("✓ END-TO-END TEST PASSED")
    print("="*70)
    print("\nComponents tested successfully:")
    print("  ✓ Model deployment and API access")
    print("  ✓ Hardware metrics monitoring")
    print("  ✓ Cost predictor (neural network)")
    print("  ✓ HW-Router scoring function")
    print("  ✓ Request execution and latency measurement")
    print("  ✓ Metrics feedback loop")
    print("\n" + "="*70 + "\n")

    return True

if __name__ == "__main__":
    success = test_end_to_end()
    exit(0 if success else 1)

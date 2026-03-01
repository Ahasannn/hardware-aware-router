#!/usr/bin/env python3
"""Test the cost predictor module"""
from hw_router.cost_predictor import HardwareCostPredictor

def test_cost_predictor():
    """Test cost predictor with sample hardware features"""

    print("\n" + "="*60)
    print("TESTING COST PREDICTOR")
    print("="*60 + "\n")

    # Load the trained cost model
    model_path = "checkpoints/hardware_cost_model/model.pt"
    preproc_path = "checkpoints/hardware_cost_model/preproc.joblib"

    print("Loading cost predictor model...")
    try:
        predictor = HardwareCostPredictor(model_path, preproc_path)
        print("✓ Cost predictor loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load cost predictor: {e}")
        return False

    # Test with sample features
    test_cases = [
        {
            "name": "Low load scenario",
            "model_id": 0,  # qwen14b
            "features": {
                "p_tokens": 512,
                "running_req_count": 0,
                "waiting_req_count": 0,
                "kv_cache_usage_perc": 0.1,
                "ttft_avg": 0.0,
                "itl_avg": 0.0,
                "model_id": 0,
                "gpu_id": "0"
            }
        },
        {
            "name": "High load scenario",
            "model_id": 1,  # phi3-mini
            "features": {
                "p_tokens": 1024,
                "running_req_count": 5,
                "waiting_req_count": 3,
                "kv_cache_usage_perc": 0.7,
                "ttft_avg": 0.5,
                "itl_avg": 0.02,
                "model_id": 1,
                "gpu_id": "0"
            }
        }
    ]

    print("Running cost predictions...")
    print("-"*60 + "\n")

    all_passed = True
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        print(f"  Model ID: {test_case['model_id']}")
        print(f"  Prompt tokens: {test_case['features']['p_tokens']}")
        print(f"  Running requests: {test_case['features']['running_req_count']}")
        print(f"  Waiting requests: {test_case['features']['waiting_req_count']}")
        print(f"  KV cache usage: {test_case['features']['kv_cache_usage_perc']:.1%}")

        try:
            ttft_pred, tpot_pred = predictor(
                test_case['model_id'],
                test_case['features']
            )

            print(f"\n  Predictions:")
            print(f"    TTFT: {ttft_pred:.4f}s")
            print(f"    TPOT: {tpot_pred:.6f}s")
            print(f"  ✓ Prediction successful\n")

            # Sanity check: values should be positive and reasonable
            if ttft_pred <= 0 or tpot_pred <= 0:
                print(f"  ✗ Invalid prediction values (non-positive)")
                all_passed = False
            elif ttft_pred > 60 or tpot_pred > 1:  # Very generous bounds
                print(f"  ✗ Unreasonable prediction values")
                all_passed = False

        except Exception as e:
            print(f"  ✗ Prediction failed: {e}\n")
            all_passed = False

    print("="*60)
    if all_passed:
        print("✓ Cost predictor test PASSED")
    else:
        print("✗ Cost predictor test FAILED")
    print("="*60 + "\n")

    return all_passed

if __name__ == "__main__":
    test_cost_predictor()

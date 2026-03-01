#!/usr/bin/env python3
"""Test basic inference from deployed models"""
import requests
import json

def test_model_inference(base_url, model_name):
    """Test a single inference request"""
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        answer = data["choices"][0]["message"]["content"]
        print(f"✓ {model_name} inference successful")
        print(f"  Response: {answer.strip()[:80]}...")
        return True
    except Exception as e:
        print(f"✗ {model_name} inference failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("TESTING MODEL INFERENCE")
    print("="*60 + "\n")

    models = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "http://localhost:8020"),
        ("Qwen/Qwen1.5-0.5B", "http://localhost:8021")
    ]

    results = []
    for model_name, base_url in models:
        print(f"\nTesting {model_name}...")
        success = test_model_inference(base_url, model_name)
        results.append((model_name, success))
        print()

    print("="*60)
    print("INFERENCE TEST RESULTS")
    print("="*60)

    for model_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_name}")

    all_passed = all(s for _, s in results)
    print("\n" + ("="*60))
    if all_passed:
        print("✓ All inference tests PASSED")
    else:
        print("✗ Some inference tests FAILED")
    print("="*60 + "\n")

    return all_passed

if __name__ == "__main__":
    main()

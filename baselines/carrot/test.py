from carrot import load_carrot_router

# Load trained router
router = load_carrot_router(
    model_dir='../../checkpoints/carrot',
    model_type='linear'  # or 'knn'
)

price = {
    "Mistral-7B-Instruct-v0.3": 0.02 / 100000
}
# Get predictions for a specific LLM
query = "What is the capital of France?"
embedding = router.encode(query)
quality = router.get_quality(embedding, "Mistral-7B-Instruct-v0.3")
baseline_cost = router.get_cost(embedding, "Mistral-7B-Instruct-v0.3") * price['Mistral-7B-Instruct-v0.3']

our_cost = baseline_cost + slo()

print(f"Quality: {quality:.4f}, Cost: {cost:.1f}")

score = qulity 
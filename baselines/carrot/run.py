"""
CARROT Training and Evaluation Script

Combined script for training and evaluating CARROT baselines.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from carrot import (
    load_and_align_data,
    filter_predictions_to_models,
    CarrotKNNBaseline,
    CarrotLinearBaseline,
    route_baseline
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train and Evaluate CARROT baselines')

    # Mode
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'both'],
                        help='Mode: train, eval, or both')

    # Data paths
    parser.add_argument('--train_dir', type=str, default='../../data_quality/train',
                        help='Directory containing training CSV files')
    parser.add_argument('--eval_dir', type=str, default='../../data_quality/eval',
                        help='Directory containing evaluation CSV files')

    # Model paths
    parser.add_argument('--output_dir', type=str, default='../../checkpoints/carrot',
                        help='Directory to save/load trained models')
    parser.add_argument('--results_dir', type=str, default='../../results/carrot',
                        help='Directory to save evaluation results')

    # Model settings
    parser.add_argument('--encoder_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Sentence transformer model for embeddings')
    parser.add_argument('--n_neighbors', type=int, default=256,
                        help='Number of neighbors for KNN')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def train(args):
    """Train CARROT models."""
    print("\n" + "="*60)
    print("TRAINING CARROT MODELS")
    print("="*60)

    # Set random seed
    np.random.seed(args.seed)

    # Load and align training data
    embeddings, quality_scores, token_counts, model_names, _ = load_and_align_data(
        args.train_dir, args.encoder_model
    )

    print(f"\n=== Training Data ===")
    print(f"Training samples: {len(embeddings)}")
    print(f"Models: {model_names}")

    # Train CARROT-KNN
    print("\n=== Training CARROT-KNN ===")
    knn_dir = os.path.join(args.output_dir, "carrot_knn")
    carrot_knn = CarrotKNNBaseline(
        n_neighbors_score=args.n_neighbors,
        n_neighbors_count=args.n_neighbors,
        metric="cosine"
    )
    carrot_knn.fit(
        embedding_train=embeddings,
        quality_train=quality_scores,
        token_count_train=token_counts,
        save_dir=knn_dir
    )

    # Train CARROT-Linear
    print("\n=== Training CARROT-Linear ===")
    linear_dir = os.path.join(args.output_dir, "carrot_linear")
    carrot_linear = CarrotLinearBaseline(fit_intercept=True)
    carrot_linear.fit(
        embedding_train=embeddings,
        quality_train=quality_scores,
        token_count_train=token_counts,
        save_dir=linear_dir
    )

    # Save metadata
    print("\n=== Saving Metadata ===")
    metadata = {
        'model_names': model_names,
        'n_train': len(embeddings),
        'encoder_model': args.encoder_model,
        'embedding_dim': embeddings.shape[1],
        'n_neighbors': args.n_neighbors,
        'seed': args.seed
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ CARROT training complete!")
    print(f"📁 Models saved to: {args.output_dir}")


def evaluate_model(model, embedding_test, quality_test, token_count_test,
                   model_names, model_type, output_dir, trained_model_names):
    """Evaluate a CARROT model."""
    print(f"\n=== Evaluating {model_type} ===")

    # Predict on test set
    Y_hat_score, Y_hat_count = model.predict(embedding_test)

    # Filter predictions to match eval models
    Y_hat_score = filter_predictions_to_models(Y_hat_score, trained_model_names, model_names)
    Y_hat_count = filter_predictions_to_models(Y_hat_count, trained_model_names, model_names)

    print(f"Filtered predictions from {len(trained_model_names)} to {len(model_names)} models")

    # Compute metrics per model
    print(f"\n{model_type} - Quality Prediction Metrics:")
    print(f"{'Model':<40} {'MSE':>10} {'MAE':>10} {'R²':>10}")
    print("=" * 72)

    quality_metrics = {}
    for i, name in enumerate(model_names):
        mse = mean_squared_error(quality_test[:, i], Y_hat_score[:, i])
        mae = mean_absolute_error(quality_test[:, i], Y_hat_score[:, i])
        r2 = r2_score(quality_test[:, i], Y_hat_score[:, i])
        print(f"{name:<40} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
        quality_metrics[name] = {'mse': mse, 'mae': mae, 'r2': r2}

    print(f"\n{model_type} - Token Count Prediction Metrics:")
    print(f"{'Model':<40} {'MSE':>10} {'MAE':>10} {'R²':>10}")
    print("=" * 72)

    cost_metrics = {}
    for i, name in enumerate(model_names):
        mse = mean_squared_error(token_count_test[:, i], Y_hat_count[:, i])
        mae = mean_absolute_error(token_count_test[:, i], Y_hat_count[:, i])
        r2 = r2_score(token_count_test[:, i], Y_hat_count[:, i])
        print(f"{name:<40} {mse:>10.1f} {mae:>10.1f} {r2:>10.4f}")
        cost_metrics[name] = {'mse': mse, 'mae': mae, 'r2': r2}

    # Routing evaluation
    print(f"\n=== Routing Evaluation ({model_type}) ===")

    # Extract model sizes (use output_tokens as proxy)
    sizes_vec = token_count_test.mean(axis=0)

    # Lambda range for cost-quality tradeoff
    lamb_range = np.linspace(0, 1, 50)

    # Compute routing performance
    router_cost, router_perf = route_baseline(
        Y_hat_score=Y_hat_score,
        Y_hat_count=Y_hat_count,
        Y_score_true=quality_test,
        Y_count_true=token_count_test,
        lamb_range=lamb_range,
        sizes_vec=sizes_vec
    )

    # Plot cost-quality tradeoff
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(router_cost, router_perf, 'o-', linewidth=2, markersize=4, label=model_type)
    plt.xlabel('Average Cost (tokens)', fontsize=12)
    plt.ylabel('Average Quality Score', fontsize=12)
    plt.title(f'{model_type} - Cost-Quality Tradeoff', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'cost_quality_{model_type.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Saved plot to: {plot_path}")

    return {
        'quality_metrics': quality_metrics,
        'cost_metrics': cost_metrics,
        'router_cost': router_cost.tolist(),
        'router_perf': router_perf.tolist(),
        'lamb_range': lamb_range.tolist()
    }


def evaluate(args):
    """Evaluate CARROT models."""
    print("\n" + "="*60)
    print("EVALUATING CARROT MODELS")
    print("="*60)

    # Load metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nLoaded metadata from: {metadata_path}")
    print(f"  Training models: {metadata['model_names']}")
    print(f"  Encoder: {metadata['encoder_model']}")

    # Load and align evaluation data
    embeddings, quality_scores, token_counts, model_names, _ = load_and_align_data(
        args.eval_dir, args.encoder_model
    )

    print(f"\n=== Evaluation Data ===")
    print(f"Evaluation samples: {len(embeddings)}")
    print(f"Evaluation models: {model_names}")

    # Load models
    knn_dir = os.path.join(args.output_dir, "carrot_knn")
    linear_dir = os.path.join(args.output_dir, "carrot_linear")

    carrot_knn = CarrotKNNBaseline(load_dir=knn_dir)
    carrot_linear = CarrotLinearBaseline(load_dir=linear_dir)

    # Evaluate both models
    results = {}

    results['CARROT-KNN'] = evaluate_model(
        carrot_knn, embeddings, quality_scores, token_counts,
        model_names, 'CARROT-KNN', args.results_dir, metadata['model_names']
    )

    results['CARROT-Linear'] = evaluate_model(
        carrot_linear, embeddings, quality_scores, token_counts,
        model_names, 'CARROT-Linear', args.results_dir, metadata['model_names']
    )

    # Save results
    results_path = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {results_path}")


def main():
    args = parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'both':
        train(args)
        evaluate(args)


if __name__ == "__main__":
    main()
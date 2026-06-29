"""
Offline Evaluation Script for Hybrid Recommender System
========================================================
Evaluates Content-Based, Collaborative Filtering, and Hybrid approaches
using a leave-k-out protocol on the interaction matrix.

Metrics: Precision@K, Recall@K, NDCG@K, Hit Rate@K

Usage:
    python evaluate.py
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from content_based_filtering import transform_data
from data_cleaning import data_for_content_filtering
import time


# ============================================================
# CONFIGURATION
# ============================================================
K = 10  # Number of recommendations to evaluate
TEST_RATIO = 0.2  # Fraction of interactions to hold out per user
MIN_INTERACTIONS = 10  # Only evaluate users with at least this many interactions
NUM_EVAL_USERS = 500  # Sample this many users for evaluation (increase for more precise results, decrease for speed)
RANDOM_SEED = 42


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def precision_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / k


def recall_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in top-k recommendations."""
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Uses binary relevance (1 if item is in test set, 0 otherwise).
    """
    top_k = recommended[:k]
    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because position is 1-indexed

    # Ideal DCG (all relevant items ranked at the top)
    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """1 if at least one relevant item is in top-k, else 0."""
    top_k = recommended[:k]
    return 1.0 if len(set(top_k) & relevant) > 0 else 0.0


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

def train_test_split_interactions(interaction_matrix: csr_matrix,
                                  test_ratio: float,
                                  min_interactions: int,
                                  seed: int):
    """
    Leave-k-out split: for each eligible user, hold out a fraction of
    their interactions as the test set.

    Returns:
        train_matrix: sparse matrix with test interactions removed
        test_dict: {user_idx: set of held-out track indices}
        eligible_users: array of user indices used for evaluation
    """
    rng = np.random.default_rng(seed)

    # Work with CSC format (columns = users)
    csc = interaction_matrix.tocsc()

    # Find users with enough interactions
    user_nnz = np.diff(csc.indptr)
    eligible_users = np.where(user_nnz >= min_interactions)[0]
    print(f"Users with >= {min_interactions} interactions: {len(eligible_users)}")

    # Build train matrix (copy)
    train_csc = csc.copy()
    test_dict = {}

    for user_idx in eligible_users:
        start, end = csc.indptr[user_idx], csc.indptr[user_idx + 1]
        track_indices = csc.indices[start:end]

        # Hold out test_ratio of interactions
        n_test = max(1, int(len(track_indices) * test_ratio))
        test_items = rng.choice(track_indices, size=n_test, replace=False)
        test_dict[user_idx] = set(test_items.tolist())

        # Zero out test interactions in train matrix
        for item_idx in test_items:
            train_csc[item_idx, user_idx] = 0.0

    train_csc.eliminate_zeros()
    train_matrix = train_csc.tocsr()

    return train_matrix, test_dict, eligible_users


# ============================================================
# RECOMMENDATION FUNCTIONS (for evaluation)
# ============================================================

def recommend_content_based(track_idx: int, transformed_data=None, k: int = 10,
                            exclude_indices: set = None, **kwargs) -> np.ndarray:
    """
    Get top-k content-based recommendations for a given track.
    Returns array of recommended track indices.
    """
    input_vector = transformed_data[track_idx].reshape(1, -1)
    scores = cosine_similarity(input_vector, transformed_data).ravel()

    # Exclude the input track itself
    scores[track_idx] = -1

    # Exclude items if specified
    if exclude_indices:
        for idx in exclude_indices:
            scores[idx] = -1

    top_k_indices = np.argsort(scores)[-k:][::-1]
    return top_k_indices


def recommend_collaborative(track_idx: int, interaction_matrix_param: csr_matrix = None,
                            k: int = 10, exclude_indices: set = None,
                            **kwargs) -> np.ndarray:
    """
    Get top-k collaborative filtering recommendations for a given track.
    Uses item-item cosine similarity on the interaction matrix.
    Returns array of recommended track indices.
    """
    interaction_matrix = interaction_matrix_param
    input_vector = interaction_matrix[track_idx]
    scores = cosine_similarity(input_vector, interaction_matrix).ravel()

    # Exclude the input track
    scores[track_idx] = -1

    if exclude_indices:
        for idx in exclude_indices:
            scores[idx] = -1

    top_k_indices = np.argsort(scores)[-k:][::-1]
    return top_k_indices


def recommend_hybrid(track_idx: int, transformed_data=None,
                     interaction_matrix_param: csr_matrix = None, k: int = 10,
                     content_weight: float = 0.5,
                     exclude_indices: set = None, **kwargs) -> np.ndarray:
    """
    Get top-k hybrid recommendations combining content-based and collaborative.
    Normalizes scores before weighted combination.
    """
    interaction_matrix = interaction_matrix_param

    # Content-based scores
    input_content = transformed_data[track_idx].reshape(1, -1)
    content_scores = cosine_similarity(input_content, transformed_data).ravel()

    # Collaborative scores
    input_collab = interaction_matrix[track_idx]
    collab_scores = cosine_similarity(input_collab, interaction_matrix).ravel()

    # Normalize both to [0, 1]
    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    content_norm = normalize(content_scores)
    collab_norm = normalize(collab_scores)

    # Weighted combination
    hybrid_scores = content_weight * content_norm + (1 - content_weight) * collab_norm

    # Exclude input track
    hybrid_scores[track_idx] = -1

    if exclude_indices:
        for idx in exclude_indices:
            hybrid_scores[idx] = -1

    top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
    return top_k_indices


# ============================================================
# EVALUATION LOOP
# ============================================================

def evaluate_approach(approach_fn, eval_users, test_dict,
                      interaction_matrix, k, **kwargs):
    """
    Evaluate a recommendation approach across sampled users.

    For each user:
    1. Pick a random track from their training interactions (seed track)
    2. Get top-k recommendations using the approach
    3. Check how many of their held-out tracks appear in recommendations
    """
    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []

    # Convert to CSC for user-level access
    csc_matrix = interaction_matrix.tocsc()

    for user_idx in eval_users:
        # Get user's training interactions (tracks they've listened to in train)
        start, end = csc_matrix.indptr[user_idx], csc_matrix.indptr[user_idx + 1]
        train_tracks = csc_matrix.indices[start:end]

        if len(train_tracks) == 0:
            continue

        # Pick a seed track (random from training set)
        seed_track = np.random.choice(train_tracks)

        # Get held-out items for this user
        relevant_items = test_dict[user_idx]

        # Get recommendations
        recommended = approach_fn(track_idx=seed_track, k=k, **kwargs)

        # Compute metrics
        precisions.append(precision_at_k(recommended, relevant_items, k))
        recalls.append(recall_at_k(recommended, relevant_items, k))
        ndcgs.append(ndcg_at_k(recommended, relevant_items, k))
        hit_rates.append(hit_rate_at_k(recommended, relevant_items, k))

    return {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "NDCG@K": np.mean(ndcgs),
        "Hit Rate@K": np.mean(hit_rates),
        "Users Evaluated": len(precisions)
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("HYBRID RECOMMENDER SYSTEM - OFFLINE EVALUATION")
    print("=" * 60)
    print(f"\nConfig: K={K}, test_ratio={TEST_RATIO}, "
          f"min_interactions={MIN_INTERACTIONS}, "
          f"eval_users={NUM_EVAL_USERS}\n")

    # Load data
    print("Loading data...")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    songs_data = pd.read_csv("data/collab_filtered_data.csv")
    track_ids = np.load("data/track_ids.npy", allow_pickle=True)
    print(f"  Interaction matrix: {interaction_matrix.shape} "
          f"({interaction_matrix.nnz} interactions)")
    print(f"  Songs in filtered data: {len(songs_data)}")

    # Prepare content features (for the filtered subset used in hybrid)
    print("\nPreparing content features...")
    content_features_data = data_for_content_filtering(songs_data)
    transformed_data = transform_data(content_features_data)
    print(f"  Transformed feature matrix shape: {transformed_data.shape}")

    # Train/test split
    print("\nCreating train/test split...")
    train_matrix, test_dict, eligible_users = train_test_split_interactions(
        interaction_matrix, TEST_RATIO, MIN_INTERACTIONS, RANDOM_SEED
    )
    print(f"  Train matrix: {train_matrix.shape} ({train_matrix.nnz} interactions)")
    print(f"  Test users: {len(test_dict)}")

    # Sample users for evaluation
    rng = np.random.default_rng(RANDOM_SEED)
    eval_users = rng.choice(eligible_users,
                            size=min(NUM_EVAL_USERS, len(eligible_users)),
                            replace=False)
    print(f"  Evaluating on {len(eval_users)} sampled users\n")

    # Evaluate Content-Based Filtering
    print("-" * 40)
    print("Evaluating: Content-Based Filtering")
    print("-" * 40)
    start_time = time.time()
    content_results = evaluate_approach(
        approach_fn=recommend_content_based,
        eval_users=eval_users,
        test_dict=test_dict,
        interaction_matrix=train_matrix,
        k=K,
        transformed_data=transformed_data
    )
    content_time = time.time() - start_time
    print(f"  Time: {content_time:.1f}s")
    for metric, value in content_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    # Evaluate Collaborative Filtering
    print("\n" + "-" * 40)
    print("Evaluating: Collaborative Filtering (Cosine)")
    print("-" * 40)
    start_time = time.time()
    collab_results = evaluate_approach(
        approach_fn=recommend_collaborative,
        eval_users=eval_users,
        test_dict=test_dict,
        interaction_matrix=train_matrix,
        k=K,
        interaction_matrix_param=train_matrix
    )
    collab_time = time.time() - start_time
    print(f"  Time: {collab_time:.1f}s")
    for metric, value in collab_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    # Evaluate Hybrid (multiple weights)
    hybrid_weights = [0.3, 0.5, 0.7]
    hybrid_results_all = {}

    for weight in hybrid_weights:
        print("\n" + "-" * 40)
        print(f"Evaluating: Hybrid (content_weight={weight})")
        print("-" * 40)
        start_time = time.time()
        hybrid_results = evaluate_approach(
            approach_fn=recommend_hybrid,
            eval_users=eval_users,
            test_dict=test_dict,
            interaction_matrix=train_matrix,
            k=K,
            transformed_data=transformed_data,
            interaction_matrix_param=train_matrix,
            content_weight=weight
        )
        hybrid_time = time.time() - start_time
        hybrid_results_all[weight] = hybrid_results
        print(f"  Time: {hybrid_time:.1f}s")
        for metric, value in hybrid_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")

    # Summary Table
    print("\n\n" + "=" * 60)
    print(f"SUMMARY OF RESULTS (K={K})")
    print("=" * 60)
    print(f"\n{'Approach':<35} {'Prec@K':>8} {'Rec@K':>8} {'NDCG@K':>8} {'Hit Rate':>8}")
    print("-" * 75)

    print(f"{'Content-Based':<35} "
          f"{content_results['Precision@K']:>8.4f} "
          f"{content_results['Recall@K']:>8.4f} "
          f"{content_results['NDCG@K']:>8.4f} "
          f"{content_results['Hit Rate@K']:>8.4f}")

    print(f"{'Collaborative (Cosine)':<35} "
          f"{collab_results['Precision@K']:>8.4f} "
          f"{collab_results['Recall@K']:>8.4f} "
          f"{collab_results['NDCG@K']:>8.4f} "
          f"{collab_results['Hit Rate@K']:>8.4f}")

    for weight, results in hybrid_results_all.items():
        label = f"Hybrid (content={weight:.1f}, collab={1-weight:.1f})"
        print(f"{label:<35} "
              f"{results['Precision@K']:>8.4f} "
              f"{results['Recall@K']:>8.4f} "
              f"{results['NDCG@K']:>8.4f} "
              f"{results['Hit Rate@K']:>8.4f}")

    # Find best hybrid weight
    best_weight = max(hybrid_results_all,
                      key=lambda w: hybrid_results_all[w]['NDCG@K'])
    best_ndcg = hybrid_results_all[best_weight]['NDCG@K']

    print(f"\n{'Best Configuration:':<35} "
          f"Hybrid (content={best_weight}, collab={1-best_weight})")
    print(f"{'Improvement over Content-Based:':<35} "
          f"{((best_ndcg - content_results['NDCG@K']) / max(content_results['NDCG@K'], 1e-10)) * 100:>+.1f}% NDCG@K")
    print(f"{'Improvement over Collaborative:':<35} "
          f"{((best_ndcg - collab_results['NDCG@K']) / max(collab_results['NDCG@K'], 1e-10)) * 100:>+.1f}% NDCG@K")
    print()


if __name__ == "__main__":
    main()

# 🎵 Hybrid Spotify Recommendation System

A hybrid music recommendation system that combines content-based filtering (audio features + tags) with item-based collaborative filtering (user listening patterns) to deliver personalized song recommendations.

## Results

Evaluated on 20,000 users using leave-k-out protocol:

| Approach | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
|----------|-------------|-----------|---------|-------------|
| Content-Based (TF-IDF + audio) | 0.0070 | 0.0162 | 0.0181 | 0.0600 |
| Collaborative (Cosine Similarity) | 0.0334 | 0.0969 | 0.0731 | 0.2540 |
| **Hybrid (0.5 / 0.5)** | **0.0344** | **0.1026** | **0.0799** | **0.2580** |

**Best configuration:** Hybrid with equal weighting achieves **9.2% NDCG@10 improvement** over collaborative-only baseline.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Input Song                         │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼─────────┐  ┌─────────▼─────────┐
│  Content-Based     │  │  Collaborative     │
│  (TF-IDF + audio   │  │  (Item-item cosine │
│   features)        │  │   on 960K users)   │
└─────────┬──────────┘  └─────────┬──────────┘
          │                       │
          │    Normalize [0,1]    │
          └───────────┬───────────┘
                      │
              Weighted Combination
                      │
          ┌───────────▼───────────┐
          │   Top-K Recommendations│
          └───────────────────────┘
```

## Tech Stack

- **ML:** scikit-learn, scipy, numpy, pandas
- **Feature Engineering:** TF-IDF, One-Hot Encoding, MinMax/Standard scaling via ColumnTransformer
- **Large-scale Processing:** Dask (9.7M interactions, 960K users)
- **Data Versioning:** DVC + AWS S3
- **App:** Streamlit
- **Deployment:** Docker, GitHub Actions CI/CD, AWS (ECR, EC2, CodeDeploy)

## Dataset

- **Music Info:** 50K+ tracks with audio features (danceability, energy, tempo, etc.) and tags
- **User Listening History:** 9.7M interactions across 960K users and 30K tracks

## How It Works

1. **Content-Based Filtering:** Transforms song features (audio attributes + TF-IDF on tags) into vectors, computes cosine similarity between songs.

2. **Collaborative Filtering:** Builds a sparse user-item interaction matrix from listening history, computes item-item cosine similarity based on shared user patterns.

3. **Hybrid Combination:** Normalizes both similarity scores to [0,1] and combines with adjustable weights. Users control the diversity-personalization tradeoff via a slider.

4. **Evaluation:** Leave-k-out protocol — hold out 20% of each user's interactions, recommend based on remaining 80%, measure if hidden songs appear in top-K.

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Pull data (requires DVC + AWS credentials)
dvc pull

# Run the app
streamlit run app.py

# Run evaluation
python evaluate.py
```

## Project Structure

```
├── app.py                      # Streamlit web application
├── content_based_filtering.py  # Content-based recommendation logic
├── collaborative_filtering.py  # Collaborative filtering + interaction matrix
├── hybrid_recommendations.py   # Hybrid recommender class
├── evaluate.py                 # Offline evaluation (Precision, Recall, NDCG, Hit Rate)
├── data_cleaning.py            # Data preprocessing pipeline
├── transform_filtered_data.py  # Feature transformation for hybrid
├── dvc.yaml                    # DVC pipeline definition
├── Dockerfile                  # Container configuration
├── .github/workflows/ci.yaml   # CI/CD pipeline
└── data/                       # Datasets (tracked via DVC)
```

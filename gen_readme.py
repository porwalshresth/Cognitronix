import os

metrics = {
    "Precision@10": 0.646,
    "Recall@10": 0.341,
    "NDCG@10": 0.727
}

readme_content = f"""# Movie Recommendation System (Dictionary Learning)

This project implements a Collaborative Filtering recommendation engine using **MiniBatch Dictionary Learning** on the MovieLens 100k dataset.

## 📊 Model Performance
| Metric | Value |
| :--- | :--- |
| **Precision@10** | {metrics['Precision@10']} |
| **Recall@10** | {metrics['Recall@10']} |
| **NDCG@10** | {metrics['NDCG@10']} |

## ⚙️ Pipeline
1. **Matrix Construction**: Pivot ratings into a User-Item matrix.
2. **Dictionary Learning**: Decompose matrix into latent features via `MiniBatchDictionaryLearning`.
3. **Reconstruction**: Predict missing ratings using the learned dictionary.
4. **Evaluation**: Ranked using Precision, Recall, and NDCG.
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ README.md created successfully!")

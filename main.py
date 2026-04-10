import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.preprocessing import normalize

# ─── STEP 1: LOAD DATA ───
cols = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=cols)
df = df.drop('timestamp', axis=1)
print("✅ Data loaded:", df.shape)

# ─── STEP 2: BUILD USER-ITEM MATRIX ───
matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
matrix_values = matrix.values
print("✅ Matrix built:", matrix_values.shape)

# ─── STEP 3: NORMALIZE ───
matrix_norm = normalize(matrix_values)
print("✅ Matrix normalized")

# ─── STEP 4: DICTIONARY LEARNING ───
print("⏳ Training Dictionary Learning... (takes 2-3 mins)")
dl = MiniBatchDictionaryLearning(
    n_components=50,
    alpha=1,
    max_iter=100,
    random_state=42
)
sparse_codes = dl.fit_transform(matrix_norm)
dictionary = dl.components_
print("✅ Dictionary learned!")

# ─── STEP 5: RECONSTRUCT MATRIX ───
reconstructed = np.dot(sparse_codes, dictionary)
print("✅ Matrix reconstructed!")

# ─── STEP 6: GET TOP 10 RECOMMENDATIONS FOR USER 1 ───
user_id = 0
user_ratings = matrix_values[user_id]
already_rated = np.where(user_ratings > 0)[0]
predicted = reconstructed[user_id]
predicted[already_rated] = -1
top10 = np.argsort(predicted)[::-1][:10]
print(f"\n🎬 Top 10 recommendations for User 1:")
print(top10)
print("\n✅ PIPELINE COMPLETE!")

# ─── STEP 7: SHOW MOVIE NAMES ───
movies = pd.read_csv('ml-100k/u.item', sep='|', names=range(24), encoding='latin-1')
movies = movies[[0, 1]]
movies.columns = ['item_id', 'title']

print("\n🎬 Recommended Movies for User 1:")
for i, movie_id in enumerate(top10):
    title = movies[movies['item_id'] == movie_id + 1]['title'].values[0]
    print(f"{i+1}. {title}")

# ─── STEP 8: EVALUATION METRICS ───
def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0

def ndcg_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(recommended_k) if item in relevant])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / idcg if idcg > 0 else 0

print("\n📊 Evaluating metrics...")
precisions, recalls, ndcgs = [], [], []

for user_idx in range(100):
    user_ratings = matrix_values[user_idx]
    relevant_items = np.where(user_ratings >= 4)[0].tolist()

    if len(relevant_items) < 2:
        continue

    test_items = relevant_items[:len(relevant_items)//2]
    train_items = relevant_items[len(relevant_items)//2:]

    predicted = reconstructed[user_idx].copy()
    predicted[train_items] = -1

    top_k = np.argsort(predicted)[::-1][:10].tolist()

    precisions.append(precision_at_k(top_k, test_items))
    recalls.append(recall_at_k(top_k, test_items))
    ndcgs.append(ndcg_at_k(top_k, test_items))

print(f"✅ Precision@10: {np.mean(precisions):.4f}")
print(f"✅ Recall@10:    {np.mean(recalls):.4f}")
print(f"✅ NDCG@10:      {np.mean(ndcgs):.4f}")

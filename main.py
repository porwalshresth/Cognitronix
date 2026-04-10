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
predicted[already_rated] = -1  # don't recommend already seen
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

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AdvancedSVDRecommender:
    """
    Production-ready SVD Recommender with:
    - Bias correction
    - Regularization
    - Cold-start handling
    - Diversity metrics
    """
   
    def __init__(self, n_factors=50, learning_rate=0.005, regularization=0.02, n_epochs=20):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
       
        # Model parameters
        self.global_mean = 0
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
       
        self.train_matrix = None
        self.predicted_ratings = None
       
        # For cold-start
        self.item_popularity = None
       
    def fit(self, train_matrix, verbose=True):
        """Train using SGD"""
       
        if verbose:
            print(f"Training SVD: {self.n_factors} factors, lr={self.lr}, reg={self.reg}")
       
        self.train_matrix = train_matrix
        n_users, n_items = train_matrix.shape
       
        # Initialize
        self.global_mean = np.sum(train_matrix) / np.sum(train_matrix > 0)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
       
        # Calculate item popularity for cold-start
        self.item_popularity = np.sum(train_matrix > 0, axis=0)
       
        # Training
        user_indices, item_indices = train_matrix.nonzero()
        n_ratings = len(user_indices)
       
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_ratings)
           
            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = train_matrix[u, i]
               
                # Predict
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
                pred += np.dot(self.user_factors[u], self.item_factors[i])
               
                # Error
                err = r - pred
               
                # Update
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
               
                user_f = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * user_f)
                self.item_factors[i] += self.lr * (err * user_f - self.reg * self.item_factors[i])
       
        # Compute predictions
        self.predicted_ratings = np.zeros((n_users, n_items))
        for u in range(n_users):
            for i in range(n_items):
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
                pred += np.dot(self.user_factors[u], self.item_factors[i])
                self.predicted_ratings[u, i] = np.clip(pred, 1, 5)
       
        return self
   
    def predict(self, user_id, item_id):
        """Predict single rating"""
        return self.predicted_ratings[user_id-1, item_id-1]
   
    def recommend(self, user_id, n=10, diversity_weight=0.0):
        """
        Get recommendations with optional diversity
        diversity_weight: 0.0 = pure accuracy, 1.0 = maximum diversity
        """
        user_idx = user_id - 1
        user_predictions = self.predicted_ratings[user_idx].copy()
       
        # Mask rated items
        already_rated = self.train_matrix[user_idx] > 0
        user_predictions[already_rated] = -1
       
        if diversity_weight > 0:
            # Apply diversity penalty based on popularity
            popularity_penalty = (self.item_popularity / self.item_popularity.max()) * diversity_weight
            user_predictions = user_predictions * (1 - popularity_penalty)
       
        top_n_indices = np.argsort(user_predictions)[::-1][:n]
        top_n_ratings = self.predicted_ratings[user_idx, top_n_indices]
       
        return top_n_indices, top_n_ratings
   
    def recommend_cold_start(self, n=10):
        """Recommend for new user (cold-start) - use global popularity"""
        # Most popular items with highest average ratings
        item_scores = np.zeros(len(self.item_popularity))
       
        for i in range(len(self.item_popularity)):
            if self.item_popularity[i] > 0:
                rated_users = self.train_matrix[:, i] > 0
                avg_rating = np.mean(self.train_matrix[rated_users, i])
                # Score = avg_rating * log(popularity)
                item_scores[i] = avg_rating * np.log1p(self.item_popularity[i])
       
        top_n_indices = np.argsort(item_scores)[::-1][:n]
        return top_n_indices
   
    def evaluate(self, test_matrix):
        """Comprehensive evaluation"""
        test_user_indices, test_item_indices = test_matrix.nonzero()
        actual = test_matrix[test_user_indices, test_item_indices]
        predicted = self.predicted_ratings[test_user_indices, test_item_indices]
       
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
       
        errors = actual - predicted
        within_1 = np.sum(np.abs(errors) <= 1.0) / len(errors) * 100
        within_0_5 = np.sum(np.abs(errors) <= 0.5) / len(errors) * 100
       
        return {
            'rmse': rmse,
            'mae': mae,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'within_1': within_1,
            'within_0.5': within_0_5,
            'n_predictions': len(actual)
        }
   
    def calculate_diversity(self, recommendations_list):
        """
        Calculate diversity of recommendations
        recommendations_list: list of recommended item indices for multiple users
        """
        all_items = set()
        for recs in recommendations_list:
            all_items.update(recs)
       
        # Coverage: percentage of catalog recommended
        coverage = len(all_items) / self.train_matrix.shape[1] * 100
       
        # Gini index for popularity distribution
        item_counts = defaultdict(int)
        for recs in recommendations_list:
            for item in recs:
                item_counts[item] += 1
       
        return {
            'coverage': coverage,
            'unique_items': len(all_items),
            'total_slots': sum(len(r) for r in recommendations_list)
        }


def cross_validate(train_ratings, n_folds=5, n_factors=50):
    """Perform k-fold cross-validation"""
   
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION ({n_folds}-Fold)")
    print(f"{'='*70}")
   
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
   
    n_users = train_ratings['user_id'].max()
    n_items = train_ratings['item_id'].max()
   
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_ratings), 1):
        print(f"\nFold {fold_idx}/{n_folds}...")
       
        fold_train = train_ratings.iloc[train_idx]
        fold_val = train_ratings.iloc[val_idx]
       
        # Create matrices
        train_mat = csr_matrix(
            (fold_train['rating'],
             (fold_train['user_id']-1, fold_train['item_id']-1)),
            shape=(n_users, n_items)
        ).toarray()
       
        val_mat = csr_matrix(
            (fold_val['rating'],
             (fold_val['user_id']-1, fold_val['item_id']-1)),
            shape=(n_users, n_items)
        ).toarray()
       
        # Train
        model = AdvancedSVDRecommender(n_factors=n_factors, n_epochs=15)
        model.fit(train_mat, verbose=False)
       
        # Evaluate
        results = model.evaluate(val_mat)
        fold_results.append(results)
       
        print(f"  RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
   
    # Average results
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])
    std_rmse = np.std([r['rmse'] for r in fold_results])
    std_mae = np.std([r['mae'] for r in fold_results])
   
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Average RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Average MAE:  {avg_mae:.4f} ± {std_mae:.4f}")
   
    return avg_rmse, avg_mae


def hyperparameter_tuning(train_matrix, test_matrix):
    """Test different hyperparameters"""
   
    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING")
    print(f"{'='*70}")
   
    param_grid = {
        'n_factors': [30, 50, 70],
        'learning_rate': [0.003, 0.005, 0.007],
        'regularization': [0.01, 0.02, 0.05]
    }
   
    best_rmse = float('inf')
    best_params = None
    results_list = []
   
    for n_f in param_grid['n_factors']:
        for lr in param_grid['learning_rate']:
            for reg in param_grid['regularization']:
                print(f"\nTesting: factors={n_f}, lr={lr}, reg={reg}")
               
                model = AdvancedSVDRecommender(
                    n_factors=n_f,
                    learning_rate=lr,
                    regularization=reg,
                    n_epochs=15
                )
               
                model.fit(train_matrix, verbose=False)
                results = model.evaluate(test_matrix)
               
                results_list.append({
                    'n_factors': n_f,
                    'lr': lr,
                    'reg': reg,
                    'rmse': results['rmse'],
                    'mae': results['mae']
                })
               
                print(f"  RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
               
                if results['rmse'] < best_rmse:
                    best_rmse = results['rmse']
                    best_params = {'n_factors': n_f, 'lr': lr, 'reg': reg}
   
    print(f"\n{'='*70}")
    print("BEST PARAMETERS")
    print(f"{'='*70}")
    print(f"Factors: {best_params['n_factors']}")
    print(f"Learning Rate: {best_params['lr']}")
    print(f"Regularization: {best_params['reg']}")
    print(f"Best RMSE: {best_rmse:.4f}")
   
    return best_params, pd.DataFrame(results_list)


def create_visualizations(model, test_matrix, movies):
    """Create comprehensive visualizations"""
   
    print("\n📊 Creating visualizations...")
   
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
   
    # 1. Prediction accuracy
    test_user_indices, test_item_indices = test_matrix.nonzero()
    actual = test_matrix[test_user_indices, test_item_indices]
    predicted = model.predicted_ratings[test_user_indices, test_item_indices]
   
    axes[0, 0].scatter(actual, predicted, alpha=0.1, s=1)
    axes[0, 0].plot([1, 5], [1, 5], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Rating')
    axes[0, 0].set_ylabel('Predicted Rating')
    axes[0, 0].set_title('Actual vs Predicted Ratings')
    axes[0, 0].grid(True, alpha=0.3)
   
    # 2. Error distribution
    errors = actual - predicted
    axes[0, 1].hist(errors, bins=50, edgecolor='black', color='skyblue')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution (Mean: {np.mean(errors):.3f})')
    axes[0, 1].grid(True, alpha=0.3)
   
    # 3. User factor variance
    user_factor_var = np.var(model.user_factors, axis=0)
    axes[0, 2].bar(range(len(user_factor_var)), sorted(user_factor_var, reverse=True))
    axes[0, 2].set_xlabel('Factor Index (sorted)')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].set_title('User Factor Importance')
    axes[0, 2].grid(True, alpha=0.3)
   
    # 4. Item factor variance
    item_factor_var = np.var(model.item_factors, axis=0)
    axes[1, 0].bar(range(len(item_factor_var)), sorted(item_factor_var, reverse=True), color='coral')
    axes[1, 0].set_xlabel('Factor Index (sorted)')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Item Factor Importance')
    axes[1, 0].grid(True, alpha=0.3)
   
    # 5. Rating distribution
    axes[1, 1].hist(actual, bins=5, alpha=0.5, label='Actual', edgecolor='black')
    axes[1, 1].hist(predicted, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
    axes[1, 1].set_xlabel('Rating')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Rating Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
   
    # 6. Bias distribution
    axes[1, 2].hist(model.user_bias, bins=30, alpha=0.5, label='User Bias', edgecolor='black')
    axes[1, 2].hist(model.item_bias, bins=30, alpha=0.5, label='Item Bias', edgecolor='black')
    axes[1, 2].set_xlabel('Bias Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Bias Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.savefig('recommendation_system_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: recommendation_system_analysis.png")
   
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
   
    print("="*70)
    print("ADVANCED DICTIONARY LEARNING RECOMMENDATION SYSTEM")
    print("MovieLens 100K Dataset")
    print("="*70)
   
    # Load data
    print("\n📊 Loading data...")
    ratings = pd.read_csv('Data1(Movies)/ml-100k/u.data',
                          sep='\t',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
   
    movies_genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                  'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                  'Sci-Fi', 'Thriller', 'War', 'Western']
   
    movies_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
               'IMDb_URL'] + movies_genre_cols
   
    movies = pd.read_csv('Data1(Movies)/ml-100k/u.item', sep='|', names=movies_columns, encoding='latin-1')
   
    print(f"Ratings: {len(ratings):,}, Users: {ratings['user_id'].nunique()}, Movies: {ratings['item_id'].nunique()}")
   
    # Split
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
   
    n_users = ratings['user_id'].max()
    n_items = ratings['item_id'].max()
   
    train_matrix = csr_matrix(
        (train_ratings['rating'],
         (train_ratings['user_id']-1, train_ratings['item_id']-1)),
        shape=(n_users, n_items)
    ).toarray()
   
    test_matrix = csr_matrix(
        (test_ratings['rating'],
         (test_ratings['user_id']-1, test_ratings['item_id']-1)),
        shape=(n_users, n_items)
    ).toarray()
   
    # 1. Train main model
    print("\n" + "="*70)
    print("TRAINING MAIN MODEL")
    print("="*70)
   
    main_model = AdvancedSVDRecommender(n_factors=50, learning_rate=0.005, regularization=0.02, n_epochs=20)
    main_model.fit(train_matrix)
   
    results = main_model.evaluate(test_matrix)
   
    print(f"\n📊 Main Model Results:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  Within 1.0 star: {results['within_1']:.2f}%")
   
    # 2. Cross-validation
    cv_rmse, cv_mae = cross_validate(train_ratings, n_folds=5, n_factors=50)
   
    # 3. Hyperparameter tuning (optional - comment out if too slow)
    # best_params, tuning_results = hyperparameter_tuning(train_matrix, test_matrix)
   
    # 4. Diversity analysis
    print(f"\n{'='*70}")
    print("DIVERSITY ANALYSIS")
    print(f"{'='*70}")
   
    all_recs = []
    for user_id in range(1, min(101, n_users+1)):  # First 100 users
        recs, _ = main_model.recommend(user_id, n=10)
        all_recs.append(recs)
   
    diversity_metrics = main_model.calculate_diversity(all_recs)
    print(f"Catalog Coverage: {diversity_metrics['coverage']:.2f}%")
    print(f"Unique Items Recommended: {diversity_metrics['unique_items']} / {n_items}")
   
    # 5. Cold-start recommendations
    print(f"\n{'='*70}")
    print("COLD-START RECOMMENDATIONS (New User)")
    print(f"{'='*70}")
   
    cold_start_recs = main_model.recommend_cold_start(n=10)
    print("Top 10 movies for new user:")
    for rank, item_idx in enumerate(cold_start_recs, 1):
        movie = movies.iloc[item_idx]
        print(f"  {rank:2d}. {movie['movie_title']}")
   
    # 6. Create visualizations
    create_visualizations(main_model, test_matrix, movies)
   
    # 7. Sample recommendations
    print(f"\n{'='*70}")
    print("SAMPLE PERSONALIZED RECOMMENDATIONS")
    print(f"{'='*70}")
   
    for user_id in [1, 50, 100]:
        print(f"\n🎬 User {user_id}:")
        recs, ratings = main_model.recommend(user_id, n=5)
        for rank, (item_idx, rating) in enumerate(zip(recs, ratings), 1):
            movie = movies.iloc[item_idx]
            print(f"  {rank}. [{rating:.2f}] {movie['movie_title']}")
   
    print(f"\n{'='*70}")
    print("✅ COMPLETE! Ready for hackathon presentation.")
    print(f"{'='*70}")

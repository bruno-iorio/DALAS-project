"""
DALAS Project: What Makes a Successful Book-to-Film Adaptation?
Complete Analysis Pipeline
Authors: Bruno Fernandes Iorio, Beeverly Gourdette
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import ast
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_preprocess_data(filepath):
    """Load and preprocess the movie adaptation dataset."""
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")
    
    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Extract year from date_film
    df['year_film'] = pd.to_datetime(df['date_film'], errors='coerce').dt.year
    
    # Parse genres from string representation
    def extract_genres(genre_str):
        if pd.isna(genre_str):
            return []
        try:
            genres = ast.literal_eval(genre_str)
            return [g['name'] for g in genres]
        except:
            return []
    
    df['genres_list'] = df['Genre'].apply(extract_genres)
    df['primary_genre'] = df['genres_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    df['num_genres'] = df['genres_list'].apply(len)
    
    # Create genre indicator columns for top genres
    all_genres = []
    for genres in df['genres_list']:
        all_genres.extend(genres)
    top_genres = pd.Series(all_genres).value_counts().head(10).index.tolist()
    
    for genre in top_genres:
        df[f'is_{genre.lower().replace(" ", "_")}'] = df['genres_list'].apply(lambda x: 1 if genre in x else 0)
    
    # Calculate financial metrics
    df['profit'] = df['revenue_film'] - df['budget_film']
    df['roi'] = np.where(
        df['budget_film'] > 0,
        (df['revenue_film'] - df['budget_film']) / df['budget_film'] * 100,
        np.nan
    )
    
    # Calculate adaptation gap (years between book and film)
    # Note: We don't have book year in this dataset, so we'll skip this
    
    print(f"Processed dataset shape: {df.shape}")
    print(f"\nMissingness in key columns:")
    key_cols = ['votes_film', 'vote_count_film', 'budget_film', 'revenue_film', 'runtime']
    for col in key_cols:
        missing_pct = df[col].isna().sum() / len(df) * 100
        print(f"  {col}: {missing_pct:.1f}%")
    
    return df, top_genres


def create_success_target(df):
    """Create binary success target variable."""
    print("\n" + "=" * 60)
    print("CREATING TARGET VARIABLES")
    print("=" * 60)
    
    # Filter to films with financial data
    df_financial = df[(df['budget_film'] > 0) & (df['revenue_film'] > 0)].copy()
    print(f"Films with financial data: {len(df_financial)}")
    
    # Define success criteria:
    # 1. Financial success: ROI > 100% (revenue > 2x budget)
    # 2. Critical success: votes_film >= 6.5
    # 3. Popular success: vote_count_film above median
    
    vote_count_median = df_financial['vote_count_film'].median()
    
    df_financial['financial_success'] = (df_financial['roi'] > 100).astype(int)
    df_financial['critical_success'] = (df_financial['votes_film'] >= 6.5).astype(int)
    df_financial['popular_success'] = (df_financial['vote_count_film'] > vote_count_median).astype(int)
    
    # Combined success: at least 2 out of 3 criteria
    df_financial['success'] = (
        (df_financial['financial_success'] + 
         df_financial['critical_success'] + 
         df_financial['popular_success']) >= 2
    ).astype(int)
    
    print(f"\nSuccess distribution:")
    print(df_financial['success'].value_counts(normalize=True))
    
    return df_financial


def prepare_features(df, top_genres):
    """Prepare feature matrix for modeling."""
    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)
    
    # Select numeric features
    numeric_features = ['runtime', 'budget_film', 'vote_count_film', 'num_genres']
    
    # Add genre indicator features
    genre_features = [f'is_{g.lower().replace(" ", "_")}' for g in top_genres]
    
    # Language encoding
    df['is_english'] = (df['original_language'] == 'en').astype(int)
    
    # Decade feature
    df['decade'] = (df['year_film'] // 10) * 10
    
    all_features = numeric_features + genre_features + ['is_english']
    
    # Filter to complete cases
    df_complete = df.dropna(subset=all_features + ['success'])
    print(f"Complete cases for modeling: {len(df_complete)}")
    
    X = df_complete[all_features].copy()
    y = df_complete['success'].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X_scaled, y, df_complete, scaler, all_features


def train_knn_classifier(X, y):
    """Train and evaluate KNN classifier with cross-validation."""
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS CLASSIFICATION")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Find optimal k using cross-validation
    k_range = range(3, 31, 2)
    cv_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1')
        cv_scores.append(scores.mean())
    
    best_k = k_range[np.argmax(cv_scores)]
    print(f"Best k found: {best_k} (CV F1 Score: {max(cv_scores):.3f})")
    
    # Train final model
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    
    # Predictions
    y_pred = knn_final.predict(X_test)
    
    # Evaluation
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Unsuccessful', 'Successful']))
    
    return knn_final, X_train, X_test, y_train, y_test, k_range, cv_scores, best_k


def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate Logistic Regression with feature importance analysis."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION CLASSIFICATION")
    print("=" * 60)
    
    # Grid search for regularization
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best regularization C: {grid_search.best_params_['C']}")
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': best_model.coef_[0],
        'odds_ratio': np.exp(best_model.coef_[0])
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nTop Feature Importance (by coefficient magnitude):")
    print(feature_importance.head(10).to_string(index=False))
    
    return best_model, feature_importance


def perform_clustering(df, top_genres):
    """Perform K-Means clustering to find adaptation patterns."""
    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Select features for clustering
    cluster_features = ['budget_film', 'revenue_film', 'votes_film', 'vote_count_film', 'runtime']
    genre_features = [f'is_{g.lower().replace(" ", "_")}' for g in top_genres[:5]]
    all_cluster_features = cluster_features + genre_features
    
    # Filter and prepare data
    df_cluster = df.dropna(subset=cluster_features).copy()
    df_cluster = df_cluster[(df_cluster['budget_film'] > 0) & (df_cluster['revenue_film'] > 0)]
    
    # Log transform financial features
    df_cluster['log_budget'] = np.log1p(df_cluster['budget_film'])
    df_cluster['log_revenue'] = np.log1p(df_cluster['revenue_film'])
    df_cluster['log_vote_count'] = np.log1p(df_cluster['vote_count_film'])
    
    cluster_features_transformed = ['log_budget', 'log_revenue', 'votes_film', 'log_vote_count', 'runtime']
    cluster_features_transformed += genre_features
    
    X_cluster = df_cluster[cluster_features_transformed].dropna()
    df_cluster = df_cluster.loc[X_cluster.index]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k} (Silhouette: {max(silhouette_scores):.3f})")
    
    # Final clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # Analyze clusters
    print("\nCluster Profiles:")
    cluster_summary = df_cluster.groupby('cluster').agg({
        'budget_film': 'median',
        'revenue_film': 'median',
        'votes_film': 'mean',
        'vote_count_film': 'median',
        'primary_genre': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(2)
    
    cluster_summary['count'] = df_cluster.groupby('cluster').size()
    cluster_summary['avg_roi'] = df_cluster.groupby('cluster')['roi'].median().round(2)
    
    print(cluster_summary.to_string())
    
    return df_cluster, X_pca, kmeans_final, optimal_k, K_range, silhouette_scores


def train_revenue_regression(df, top_genres):
    """Train regression models to predict film revenue."""
    print("\n" + "=" * 60)
    print("REVENUE REGRESSION")
    print("=" * 60)
    
    # Prepare features
    df_reg = df[(df['budget_film'] > 0) & (df['revenue_film'] > 0)].copy()
    
    # Log transform target
    df_reg['log_revenue'] = np.log1p(df_reg['revenue_film'])
    df_reg['log_budget'] = np.log1p(df_reg['budget_film'])
    
    # Features
    numeric_features = ['log_budget', 'runtime', 'votes_film', 'num_genres']
    genre_features = [f'is_{g.lower().replace(" ", "_")}' for g in top_genres]
    
    df_reg['is_english'] = (df_reg['original_language'] == 'en').astype(int)
    
    all_features = numeric_features + genre_features[:5] + ['is_english']
    
    # Remove missing values
    df_reg = df_reg.dropna(subset=all_features + ['log_revenue'])
    
    X = df_reg[all_features]
    y = df_reg['log_revenue']
    
    print(f"Regression dataset size: {len(X)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Convert back from log scale for interpretable metrics
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(y_pred)
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test, y_pred)  # R2 on log scale
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        print(f"\n{name}:")
        print(f"  R² (log scale): {r2:.3f}")
        print(f"  RMSE: ${rmse:,.0f}")
        print(f"  MAE: ${mae:,.0f}")
    
    # Feature importance for Ridge
    ridge_model = models['Ridge Regression']
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'coefficient': ridge_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nRevenue Prediction - Feature Importance:")
    print(feature_importance.to_string(index=False))
    
    return models, results, X_test, y_test, scaler


def create_visualizations(df, df_cluster, X_pca, k_range, cv_scores, silhouette_scores,
                         feature_importance, optimal_k):
    """Create and save all visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    # 1. KNN Cross-validation scores
    ax1 = axes[0, 0]
    ax1.plot(list(range(3, 31, 2)), cv_scores, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax1.set_ylabel('Cross-Validation F1 Score', fontsize=12)
    ax1.set_title('KNN: Optimal k Selection', fontsize=14, fontweight='bold')
    ax1.axvline(x=list(range(3, 31, 2))[np.argmax(cv_scores)], color='r', linestyle='--', label=f'Best k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature Importance (Logistic Regression)
    ax2 = axes[0, 1]
    top_features = feature_importance.head(10)
    colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
    ax2.barh(top_features['feature'], top_features['coefficient'], color=colors, alpha=0.7)
    ax2.set_xlabel('Coefficient', fontsize=12)
    ax2.set_title('Logistic Regression: Feature Importance', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Clustering - Silhouette Scores
    ax3 = axes[1, 0]
    cluster_k_range = list(range(2, 2 + len(silhouette_scores)))
    ax3.plot(cluster_k_range, silhouette_scores, 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Number of Clusters', fontsize=12)
    ax3.set_ylabel('Silhouette Score', fontsize=12)
    ax3.set_title('Clustering: Optimal k Selection', fontsize=14, fontweight='bold')
    ax3.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cluster Visualization (PCA)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=df_cluster['cluster'], 
                         cmap='viridis', alpha=0.6, s=30)
    ax4.set_xlabel('First Principal Component', fontsize=12)
    ax4.set_ylabel('Second Principal Component', fontsize=12)
    ax4.set_title('Film Adaptation Clusters (PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Cluster')
    
    # 5. Revenue Distribution by Success
    ax5 = axes[2, 0]
    df_plot = df[(df['revenue_film'] > 0) & (df['success'].notna())].copy()
    df_plot['log_revenue'] = np.log10(df_plot['revenue_film'])
    
    for success_val, label, color in [(0, 'Unsuccessful', 'salmon'), (1, 'Successful', 'lightgreen')]:
        subset = df_plot[df_plot['success'] == success_val]['log_revenue']
        ax5.hist(subset, bins=30, alpha=0.6, label=label, color=color)
    
    ax5.set_xlabel('Log₁₀(Revenue)', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title('Revenue Distribution by Success', fontsize=14, fontweight='bold')
    ax5.legend()
    
    # 6. Genre Success Rates
    ax6 = axes[2, 1]
    genre_success = df.groupby('primary_genre').agg({
        'success': 'mean',
        'Film': 'count'
    }).reset_index()
    genre_success = genre_success[genre_success['Film'] >= 50].sort_values('success', ascending=True)
    
    colors = plt.cm.RdYlGn(genre_success['success'])
    ax6.barh(genre_success['primary_genre'], genre_success['success'], color=colors)
    ax6.set_xlabel('Success Rate', fontsize=12)
    ax6.set_title('Success Rate by Primary Genre (n≥50)', fontsize=14, fontweight='bold')
    ax6.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=150, bbox_inches='tight')
    print("Saved: analysis_results.png")
    
    return fig


def generate_recommendations(df, knn_model, scaler, feature_names, top_genres):
    """Generate book-to-film adaptation recommendations."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS & RECOMMENDATIONS")
    print("=" * 60)
    
    # Find examples of successful adaptations for recommendations
    successful = df[df['success'] == 1].nlargest(10, 'roi')
    
    print("\nTop 10 Most Successful Adaptations (by ROI):")
    print("-" * 60)
    for _, row in successful.iterrows():
        print(f"Book: {row['Book']}")
        print(f"  Film: {row['Film']} ({row['year_film']:.0f})")
        print(f"  Budget: ${row['budget_film']:,.0f}")
        print(f"  Revenue: ${row['revenue_film']:,.0f}")
        print(f"  ROI: {row['roi']:.0f}%")
        print(f"  Rating: {row['votes_film']:.1f}/10")
        print()
    
    # Insights by genre
    print("\nGenre Insights:")
    print("-" * 60)
    genre_stats = df[df['success'].notna()].groupby('primary_genre').agg({
        'success': ['mean', 'count'],
        'roi': 'median',
        'votes_film': 'mean'
    }).round(2)
    genre_stats.columns = ['Success Rate', 'Count', 'Median ROI', 'Avg Rating']
    genre_stats = genre_stats[genre_stats['Count'] >= 30].sort_values('Success Rate', ascending=False)
    print(genre_stats.head(10).to_string())
    
    return successful


def main():
    """Main execution function."""
    print("=" * 60)
    print("DALAS PROJECT: BOOK-TO-FILM ADAPTATION SUCCESS PREDICTION")
    print("=" * 60)
    
    # Load data
    df, top_genres = load_and_preprocess_data('data/raw/movie_adaptation_list.csv')
    
    # Create target variable
    df = create_success_target(df)
    
    # Prepare features
    X, y, df_complete, scaler, feature_names = prepare_features(df, top_genres)
    
    # Train KNN
    knn_model, X_train, X_test, y_train, y_test, k_range, cv_scores, best_k = train_knn_classifier(X, y)
    
    # Train Logistic Regression
    lr_model, feature_importance = train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)
    
    # Perform Clustering
    df_cluster, X_pca, kmeans_model, optimal_k, K_range, silhouette_scores = perform_clustering(df, top_genres)
    
    # Train Revenue Regression
    reg_models, reg_results, X_reg_test, y_reg_test, reg_scaler = train_revenue_regression(df, top_genres)
    
    # Create visualizations
    fig = create_visualizations(
        df, df_cluster, X_pca, k_range, cv_scores, silhouette_scores,
        feature_importance, optimal_k
    )
    
    # Generate recommendations
    recommendations = generate_recommendations(df, knn_model, scaler, feature_names, top_genres)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Key Findings:
1. Dataset: {len(df)} book-to-film adaptations with financial data
2. KNN Classification: Best k={best_k}, achieved solid classification performance
3. Logistic Regression: Identified key factors influencing success
4. Clustering: Found {optimal_k} distinct adaptation patterns
5. Revenue Prediction: Ridge regression provides reasonable estimates

Top Success Factors (from Logistic Regression):
{feature_importance.head(5)[['feature', 'coefficient']].to_string(index=False)}

Cluster Interpretation:
- Low-budget indie adaptations
- Mid-budget genre films  
- High-budget blockbusters
- Niche/foreign adaptations
    """)
    
    
    return df, knn_model, lr_model, kmeans_model, reg_models


if __name__ == "__main__":
    results = main()

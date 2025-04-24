"""
Module de prétraitement des données pour le projet STA211.
Ce module contient des fonctions pour le chargement, le nettoyage et la gestion des valeurs manquantes.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

__all__ = [
    'clean_data',
    'load_data',
    'analyze_missing_values',
    'find_optimal_k',
    'handle_missing_values'
]

def clean_data(data):
    """Nettoie les données en enlevant les guillemets et les espaces superflus."""
    # Nettoyer les noms de colonnes
    data.columns = data.columns.str.strip().str.replace('"', '')
    
    # Nettoyer les valeurs string
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip().str.replace('"', '')
    
    return data

def load_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données
    """
    try:
        # Essayer d'abord avec le séparateur tabulation
        df = pd.read_csv(file_path, sep='\t')
        
        # Si la lecture échoue, essayer avec la virgule
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep=',')
            
        # Vérifier la présence de la colonne 'outcome'
        if 'outcome' not in df.columns:
            raise ValueError("La colonne 'outcome' est manquante dans le fichier de données")
            
        # Nettoyer les données
        df = clean_data(df)
        
        # Afficher les informations sur le dataset
        print(f"Dimensions du dataset: {df.shape}")
        print(f"Colonnes: {df.columns.tolist()}")
        print(f"Types de données: {df.dtypes}")
        print("\nAperçu des données:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        return None

def analyze_missing_values(df):
    """
    Analyze missing values in the dataset and provide detailed insights.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset to analyze
        
    Returns:
    --------
    dict
        A dictionary containing various statistics about missing values
    """
    # Calculate basic missing value statistics
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / total_cells) * 100
    
    # Get columns with missing values and their percentages
    missing_by_column = df.isnull().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]
    missing_percentages = (missing_by_column / len(df)) * 100
    
    # Group columns by percentage of missing values
    high_missing = missing_percentages[missing_percentages > 30]
    medium_missing = missing_percentages[(missing_percentages > 5) & (missing_percentages <= 30)]
    low_missing = missing_percentages[missing_percentages <= 5]
    
    # Create summary statistics
    summary = {
        'total_missing': total_missing,
        'missing_percentage': missing_percentage,
        'columns_with_missing': len(missing_by_column),
        'high_missing_cols': len(high_missing),
        'medium_missing_cols': len(medium_missing),
        'low_missing_cols': len(low_missing),
        'missing_by_column': missing_by_column.to_dict(),
        'missing_percentages': missing_percentages.to_dict()
    }
    
    # Print detailed analysis
    print("\nMissing Values Analysis:")
    print("-----------------------")
    print(f"Total missing values: {total_missing:,} ({missing_percentage:.2f}% of all data)")
    print(f"Number of columns with missing values: {len(missing_by_column)}")
    print("\nBreakdown by severity:")
    print(f"High (>30%): {len(high_missing)} columns")
    print(f"Medium (5-30%): {len(medium_missing)} columns")
    print(f"Low (≤5%): {len(low_missing)} columns")
    
    print("\nColumns with highest missing percentages:")
    print(missing_percentages.sort_values(ascending=False).head().to_string())
    
    return summary

def find_optimal_k(data, continuous_cols=['X1', 'X2', 'X3'], k_range=range(1, 21), cv_folds=5, sample_size=1000):
    """
    Find the optimal k for KNN imputation using cross-validation.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    continuous_cols : list
        List of continuous columns to use for optimization
    k_range : range
        Range of k values to test
    cv_folds : int
        Number of folds for cross-validation
    sample_size : int
        Number of samples to use for optimization (to speed up the process)
    
    Returns:
    --------
    int
        The optimal value of k
    """
    # Select only the continuous columns
    X = data[continuous_cols].copy()
    
    # If data is too large, take a random sample
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)
    
    # Create artificial missing values for validation
    np.random.seed(42)
    mask = np.random.rand(*X.shape) < 0.2
    X_missing = X.copy()
    X_missing[mask] = np.nan
    
    # Initialize arrays to store results
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mse_scores = []
    
    # Test different k values
    for k in k_range:
        fold_scores = []
        for train_idx, val_idx in cv.split(X):
            # Split data
            X_train = X_missing.iloc[train_idx]
            X_val = X_missing.iloc[val_idx]
            X_true = X.iloc[val_idx]
            
            # Fit imputer
            imputer = KNNImputer(n_neighbors=k)
            imputer.fit(X_train)
            
            # Impute validation set
            X_imputed = pd.DataFrame(
                imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            
            # Calculate MSE only for the artificially missing values
            val_mask = mask[val_idx]
            mse = mean_squared_error(
                X_true[val_mask],
                X_imputed[val_mask]
            )
            fold_scores.append(mse)
            
        mse_scores.append(np.mean(fold_scores))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, mse_scores, 'o-')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Mean Squared Error')
    plt.title('KNN Imputation Performance vs k')
    plt.grid(True)
    plt.show()
    
    # Return optimal k
    optimal_k = k_range[np.argmin(mse_scores)]
    print(f"Optimal k found: {optimal_k}")
    return optimal_k

def handle_missing_values(df, strategy='advanced', display_info=True, save_results=True, output_path='data_processed.csv'):
    """
    Handle missing values in the dataset using different strategies.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    strategy : str
        The strategy to use for imputation:
        - 'simple': use median for variables with >30% missing, mean for others
        - 'advanced': use correlation-based KNN for variables with >20% missing, mean for others
        - 'multiple': use IterativeImputer (MICE)
    display_info : bool
        Whether to display information about the process
    save_results : bool
        Whether to save the imputed data to CSV
    output_path : str
        Path where to save the processed data if save_results is True
        
    Returns:
    --------
    pandas.DataFrame
        The dataset with imputed values
    """
    # Copy the dataframe to avoid modifying the original
    df_imputed = df.copy()
    
    # Calculate missing percentages
    missing_percentages = df.isnull().mean() * 100
    
    if display_info:
        print("\nMissing values analysis:")
        print(missing_percentages[missing_percentages > 0])
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlations with target
    target_numeric = (df['outcome'] == 'ad.').astype(int)
    correlations = pd.Series({
        col: df[col].corr(target_numeric) 
        for col in numeric_cols 
        if col != 'outcome'
    })
    
    if strategy == 'simple':
        # Simple imputation
        for col in numeric_cols:
            missing_pct = missing_percentages[col]
            if missing_pct > 30:
                df_imputed[col].fillna(df[col].median(), inplace=True)
            elif missing_pct > 0:
                df_imputed[col].fillna(df[col].mean(), inplace=True)
                
    elif strategy == 'advanced':
        # Find continuous columns with high missing rates
        high_missing_cols = [col for col in numeric_cols 
                           if missing_percentages[col] > 20]
        
        if high_missing_cols:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # For each column with high missing values
            for col in high_missing_cols:
                # Find highly correlated features (|corr| > 0.5) to use for imputation
                corr_features = corr_matrix[col][abs(corr_matrix[col]) > 0.5].index.tolist()
                corr_features = [f for f in corr_features if f != col]
                
                if corr_features:
                    if display_info:
                        print(f"\nUsing correlated features for {col}: {corr_features}")
                    
                    # Find optimal k using correlated features
                    features_for_imputation = [col] + corr_features
                    optimal_k = find_optimal_k(
                        df, 
                        continuous_cols=features_for_imputation,
                        k_range=range(1, min(21, len(df) // 10))
                    )
                    
                    # Apply KNN imputation with optimal k
                    imputer = KNNImputer(n_neighbors=optimal_k)
                    df_subset = df[features_for_imputation].copy()
                    df_imputed[col] = pd.DataFrame(
                        imputer.fit_transform(df_subset),
                        columns=df_subset.columns,
                        index=df_subset.index
                    )[col]
                else:
                    # If no strong correlations, use median
                    if display_info:
                        print(f"\nNo strong correlations found for {col}, using median imputation")
                    df_imputed[col].fillna(df[col].median(), inplace=True)
        
        # Use mean imputation for remaining columns with missing values
        remaining_cols = [col for col in numeric_cols 
                         if col not in high_missing_cols 
                         and missing_percentages[col] > 0]
        
        for col in remaining_cols:
            df_imputed[col].fillna(df[col].mean(), inplace=True)
            
    elif strategy == 'multiple':
        # Multiple imputation using MICE
        imputer = IterativeImputer(
            random_state=42,
            max_iter=20,
            min_value=df[numeric_cols].min(),
            max_value=df[numeric_cols].max()
        )
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Final check for any remaining missing values
    remaining_missing = df_imputed.isnull().sum()
    if display_info and (remaining_missing > 0).any():
        print("\nWarning: Some missing values remain:")
        print(remaining_missing[remaining_missing > 0])
    
    # Save the processed data if requested
    if save_results:
        # Save all data in one file without splitting
        df_imputed.to_csv(output_path, index=False)
        if display_info:
            print(f"\nProcessed data has been saved to: {output_path}")
    
    return df_imputed 
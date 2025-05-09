
import numpy as np
import pandas as pd
from scipy.stats import boxcox

def transform_selected_variables(df):
    df_transformed = df.copy()

    # X1 : log(1 + x)
    df_transformed['X1_log'] = np.log1p(df['X1'])

    # X2 : Box-Cox
    x2_nonan = df['X2'].dropna()
    x2_transformed, _ = boxcox(x2_nonan)
    df_transformed['X2_boxcox'] = np.nan
    df_transformed.loc[x2_nonan.index, 'X2_boxcox'] = x2_transformed

    # X3 : Box-Cox
    x3_nonan = df['X3'].dropna()
    x3_transformed, _ = boxcox(x3_nonan)
    df_transformed['X3_boxcox'] = np.nan
    df_transformed.loc[x3_nonan.index, 'X3_boxcox'] = x3_transformed

    # Supprimer les colonnes originales
    df_transformed.drop(columns=['X1', 'X2', 'X3'], inplace=True)

    return df_transformed

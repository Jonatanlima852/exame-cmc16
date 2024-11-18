import pandas as pd
from typing import Dict, Any
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza inicial dos dados
    """
    df_clean = df.copy()
    # Remove linhas com valores nulos
    df_clean = df_clean.dropna()
    return df_clean

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features para o modelo
    """
    df_processed = df.copy()
    # Adicionar idade do carro
    if 'ano' in df_processed.columns:
        df_processed['idade_carro'] = 2024 - df_processed['ano']
    return df_processed

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variáveis categóricas
    """
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    
    return df_encoded 
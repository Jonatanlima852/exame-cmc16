import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

class DataPreprocessor:
    """
    Classe para pré-processamento dos dados de carros
    """
    def __init__(self):
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = StandardScaler()
        
    def knn_impute(self, df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Realiza imputação de dados usando KNN
        
        Args:
            df: DataFrame com os dados
            n_neighbors: Número de vizinhos para KNN
            
        Returns:
            DataFrame com dados imputados
        """
        df_encoded = df.copy()
        
        # Codifica variáveis categóricas para KNN
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
            
        # Aplica KNN Imputer
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(
            knn_imputer.fit_transform(df_encoded), 
            columns=df_encoded.columns
        )
        
        # Reverte codificação para variáveis categóricas
        for col in df.select_dtypes(include='object').columns:
            df_imputed[col] = df_imputed[col].round().astype(int).map(
                dict(enumerate(df[col].astype('category').cat.categories))
            )
            
        return df_imputed

    def remove_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove outliers usando método IQR
        
        Args:
            df: DataFrame com os dados
            column: Nome da coluna para remover outliers
            
        Returns:
            DataFrame sem outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria novas features
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()
        
        # Cria features compostas
        df_new['engine_transmission'] = df_new['engine'] * df_new['transmission']
        df_new['int_ext_color'] = df_new['int_col'] * df_new['ext_col']
        
        # Remove colunas originais usadas nas features compostas
        df_new.drop(columns=['engine', 'transmission', 'int_col', 'ext_col'], inplace=True)
        
        # Calcula idade do carro
        df_new['car_age'] = 2024 - df_new['model_year']
        df_new.drop(columns=['model_year'], inplace=True)
        
        return df_new

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Realiza todo o pré-processamento dos dados
        
        Args:
            df: DataFrame com os dados
            is_training: Flag indicando se é conjunto de treino
            
        Returns:
            DataFrame pré-processado
        """
        # Remove ID se existir
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        # Imputação de dados
        df_processed = self.knn_impute(df, n_neighbors=25)
        
        # Criação de features
        df_processed = self.create_features(df_processed)
        
        # Remove outliers apenas se for conjunto de treino
        if is_training:
            df_processed = self.remove_outliers_iqr(df_processed, 'milage')
            df_processed = self.remove_outliers_iqr(df_processed, 'price')
            df_processed.reset_index(drop=True, inplace=True)
        
        return df_processed

    def prepare_for_training(self, df: pd.DataFrame, target_col: str = 'price', 
                           scale_features: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados para treinamento
        
        Args:
            df: DataFrame pré-processado
            target_col: Nome da coluna alvo
            scale_features: Se deve escalonar as features numéricas
            
        Returns:
            Tuple com features (X) e target (y)
        """
        # Remove ID se existir
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        # Imputação de dados
        df_processed = self.knn_impute(df, n_neighbors=25)
        
        # Criação de features
        df_processed = self.create_features(df_processed)
        
        # Remove outliers apenas se for conjunto de treino
        if scale_features:
            df_processed = self.remove_outliers_iqr(df_processed, 'milage')
            df_processed = self.remove_outliers_iqr(df_processed, 'price')
            df_processed.reset_index(drop=True, inplace=True)
        
        # Escala features numéricas
        if scale_features:
            df_processed = pd.DataFrame(self.scaler.fit_transform(df_processed), columns=df_processed.columns)
        
        # Codifica variáveis categóricas
        df_processed = pd.DataFrame(self.ordinal_encoder.fit_transform(df_processed), columns=df_processed.columns)
        
        # Separa features e target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        return X, y 
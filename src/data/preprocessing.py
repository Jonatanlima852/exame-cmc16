import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pathlib import Path
import joblib

class DataPreprocessor:
    """
    Classe para pré-processamento dos dados de carros
    """
    def __init__(self):
        base_path = Path(__file__).parent.parent  # vai para o diretório src
        encoder_path = base_path / 'models' / 'ordinal_encoder.pkl'
        
        # Atualizando a ordem para corresponder ao CSV original
        self.feature_order = [
            'brand',
            'model',
            'car_age',
            'milage',
            'fuel_type',
            'engine_transmission',
            'int_ext_color',
            'accident',
            'clean_title'
        ]
        
        # Carrega o encoder salvo ou cria um novo se não existir
        try:
            self.ordinal_encoder = joblib.load(encoder_path)
        except:
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
        """
        print("\n=== INÍCIO DO PREPROCESSAMENTO ===")
        print("Colunas iniciais:", df.columns.tolist())
        
        df_processed = df.copy()
        
        # Remove ID se existir
        if 'id' in df_processed.columns:
            df_processed = df_processed.drop(columns=['id'])
            print("\nApós remover ID:", df_processed.columns.tolist())
            
        # Calcula car_age se necessário
        if 'model_year' in df_processed.columns:
            df_processed['car_age'] = 2024 - df_processed['model_year'].astype(int)
            df_processed = df_processed.drop(columns=['model_year'])
            print("\nApós calcular car_age:", df_processed.columns.tolist())
            
        # Imputação de dados
        df_processed = self.knn_impute(df_processed, n_neighbors=25)
        print("\nApós KNN impute:", df_processed.columns.tolist())
        
        # Codifica variáveis categóricas
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            df_processed[cat_cols] = self.ordinal_encoder.transform(df_processed[cat_cols].astype(str))
            print("\nApós codificação categórica:", df_processed.columns.tolist())
        
        # Criação de features compostas
        if 'engine' in df_processed.columns and 'transmission' in df_processed.columns:
            df_processed['engine_transmission'] = (
                df_processed.apply(
                    lambda x: f"{str(x['engine'])}_{str(x['transmission'])}", 
                    axis=1
                ).astype('category').cat.codes
            )
            df_processed = df_processed.drop(columns=['engine', 'transmission'])
            print("\nApós criar engine_transmission:", df_processed.columns.tolist())
            
        if 'int_col' in df_processed.columns and 'ext_col' in df_processed.columns:
            df_processed['int_ext_color'] = (
                df_processed.apply(
                    lambda x: f"{str(x['int_col'])}_{str(x['ext_col'])}", 
                    axis=1
                ).astype('category').cat.codes
            )
            df_processed = df_processed.drop(columns=['int_col', 'ext_col'])
            print("\nApós criar int_ext_color:", df_processed.columns.tolist())
        
        # Após todas as transformações, forçar a ordem correta das colunas
        # correct_order = [
        #     'brand',
        #     'model',
        #     'milage',
        #     'fuel_type',
        #     'accident',
        #     'clean_title',
        #     'engine_transmission',
        #     'int_ext_color',
        #     'car_age'
        # ]

        # Ordem alternativa (comentada) caso seja necessária no futuro
        correct_order = [
            'model', 
            'brand', 
            'car_age',
            'milage', 
            'fuel_type',
            'engine_transmission',
            'int_ext_color',
            'accident', 
            'clean_title'
        ]
        
        print("\nOrdem atual antes do reordenamento final:", df_processed.columns.tolist())
        
        # Verifica se todas as colunas necessárias existem
        missing_cols = set(correct_order) - set(df_processed.columns)
        if missing_cols:
            raise ValueError(f"Colunas ausentes: {missing_cols}")
        
        # Força a ordem exata das colunas
        df_processed = df_processed[correct_order]
        
        print("\nOrdem final após reordenamento:", df_processed.columns.tolist())
        print("=== FIM DO PREPROCESSAMENTO ===\n")
        
        # Antes de retornar, vamos garantir e verificar a ordem final
        df_processed = df_processed[correct_order]
        print("\nOrdem EXATA antes de retornar do preprocess_data:", df_processed.columns.tolist())
        
        # Forçar o tipo das colunas para float
        for col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(float)
        
        return df_processed

    def prepare_for_training(self, df: pd.DataFrame, target_col: str = 'price', 
                           scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados para treinamento
        
        Args:
            df: DataFrame pré-processado
            target_col: Nome da coluna alvo
            scale_features: Se deve escalonar as features numéricas
            
        Returns:
            Tuple com features (X) e target (y)
        """
        print("\n=== INÍCIO DO PREPARE FOR TRAINING ===")
        print("Colunas iniciais:", df.columns.tolist())
        
        df_processed = self.preprocess_data(df, is_training=True)
        print("\nApós preprocessamento:", df_processed.columns.tolist())
        
        # Escala features numéricas específicas
        if scale_features:
            numeric_cols = ['brand', 'model', 'milage', 'int_ext_color', 'engine_transmission']
            df_processed.loc[:, numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
            print("\nApós scaling:", df_processed.columns.tolist())
        
        # Garante a ordem correta das colunas antes de separar X e y
        expected_columns = [
            'brand', 'model', 'car_age', 'milage', 'fuel_type',
            'engine_transmission', 'int_ext_color', 'accident', 'clean_title'
        ]
        df_processed = df_processed.reindex(columns=expected_columns + [target_col])
        print("\nOrdem final antes de separar X e y:", df_processed.columns.tolist())
        
        # Separa features e target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        print("\nColunas finais de X:", X.columns.tolist())
        print("=== FIM DO PREPARE FOR TRAINING ===\n")
        
        return X, y
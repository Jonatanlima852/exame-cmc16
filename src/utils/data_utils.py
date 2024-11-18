import pandas as pd
import os

def load_data(filepath):
    """
    Carrega dados do arquivo especificado
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Erro ao carregar dados: {str(e)}")

def save_data(data, filepath):
    """
    Salva dados no arquivo especificado
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Erro ao salvar dados: {str(e)}") 
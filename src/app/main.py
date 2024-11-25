import streamlit as st
from models.model import CarPriceModel
import pandas as pd
import json
from pathlib import Path

@st.cache_data  # Cache para melhor performance
def load_brand_models():
    """Carrega o dicionário de marcas e modelos do JSON"""
    try:
        json_path = Path(__file__).parent.parent / 'models' / 'brand_models.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar dicionário de marcas/modelos: {str(e)}")
        return {"Other": ["Other"]}

def main():
    st.title('Previsão de Preço de Carros')
    
    # Carregando o modelo e dicionário de marcas/modelos
    try:
        model = CarPriceModel()
        brand_models = load_brand_models()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return

    # Formulário de entrada
    with st.form("car_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox(
                "Marca do Carro",
                options=sorted(brand_models.keys())
            )
            
            car_model = st.selectbox(
                "Modelo do Carro",
                options=brand_models.get(brand, ["Other"])
            )
            
            year = st.number_input("Ano do Modelo", min_value=1900, max_value=2024, value=2020)
            mileage = st.number_input("Quilometragem", min_value=0, value=50000)
            fuel_type = st.selectbox("Tipo de Combustível", 
                                   ["Gasoline", "Diesel", "E85 Flex Fuel"])
            
        with col2:
            engine = st.selectbox("Motor", [
                "2.0", "1.6", "3.0", "2.5", "1.8", "Other"
            ])
            
            transmission = st.selectbox("Transmissão", ["Automatic", "Manual"])
            ext_color = st.selectbox("Cor Externa", [
                "Black", "White", "Silver", "Gray", "Blue", "Red", "Other"
            ])
            int_color = st.selectbox("Cor Interna", [
                "Black", "Beige", "Gray", "Brown", "Other"
            ])
            accident = st.selectbox("Histórico de Acidentes", 
                                  ["None reported", "At least 1 accident"])
            clean_title = st.selectbox("Título Limpo", ["Yes", "No"])

        submitted = st.form_submit_button("Prever Preço")

    if submitted:
        # Preparar dados para previsão
        input_data = {
            'brand': brand,
            'model': car_model,
            'model_year': year,
            'milage': mileage,
            'fuel_type': fuel_type,
            'engine': engine,
            'transmission': transmission,
            'ext_col': ext_color,
            'int_col': int_color,
            'accident': accident,
            'clean_title': clean_title
        }

        try:
            # Fazer previsão
            prediction = model.predict(input_data)
            
            # Mostrar resultado
            st.success(f"Preço previsto: ${prediction[0]:,.2f}")
            
            # Mostrar detalhes adicionais
            st.info("""
            **Detalhes da Previsão:**
            - Este valor é uma estimativa baseada nos dados históricos
            - Considere fatores adicionais como condição geral do veículo
            - O preço real pode variar dependendo do mercado local
            """)
            
        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {str(e)}")
            st.error("Por favor, verifique se todos os campos foram preenchidos corretamente.")

if __name__ == "__main__":
    main() 
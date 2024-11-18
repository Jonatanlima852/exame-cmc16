import streamlit as st

def main():
    st.set_page_config(
        page_title="Previsão de Preços de Carros",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Previsão de Preços de Carros")
    
    st.sidebar.header("Características do Veículo")
    
    # Inputs básicos para exemplo
    marca = st.sidebar.selectbox("Marca", ["Toyota", "Volkswagen", "Ford", "Honda"])
    modelo = st.sidebar.text_input("Modelo")
    ano = st.sidebar.number_input("Ano", min_value=1990, max_value=2024)
    kilometragem = st.sidebar.number_input("Kilometragem", min_value=0)
    
    if st.sidebar.button("Fazer Previsão"):
        st.info("Funcionalidade de previsão será implementada em breve!")

if __name__ == "__main__":
    main() 
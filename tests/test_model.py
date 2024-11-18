import pytest
from src.models.model import CarPriceModel

def test_model_initialization():
    model = CarPriceModel()
    assert model is not None
    assert model.model is None

def test_model_preprocessing():
    model = CarPriceModel()
    # Adicionar mais testes conforme implementação
    pass 
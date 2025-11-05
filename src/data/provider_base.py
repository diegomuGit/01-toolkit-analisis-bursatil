from abc import ABC, abstractmethod
from src.core.price_series import PriceSeries

class DataProviderBase(ABC):
    '''Interfaz base para las API proveedoras de datos'''

    @abstractmethod
    def get_serie_precios(self, ticker: str, start:str, end: str) -> PriceSeries:
        '''Descarga la serie histórica en formato de objeto PriceSeries'''
        pass
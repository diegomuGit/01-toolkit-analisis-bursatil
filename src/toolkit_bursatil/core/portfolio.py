"""
Clase Portfolio
---------------
Representa una cartera compuesta por varios PriceSeries y sus pesos asociados.
Calcula retornos, rentabilidad media y volatilidad conjunta.
"""

from attr import dataclass, field
from toolkit_bursatil.core.price_series import PriceSeries
import pandas as pd
import numpy as np



@dataclass
class Portfolio:
    name: str
    assets: dict[str, PriceSeries]
    mean: float = field(init=False)
    std: float = field(init=False)
    returns_df : pd.DataFrame = field(init=False)
    portfolio_returns: pd.Series = field(init=False) 
    weights: dict[str, float]

    def __post_init__(self):
        '''Validar que los pesos suman 1'''
        if not sum(self.weights.values()) == 1.00:
            raise ValueError("Los pesos deben sumar 1")

        """Cálculo de estadísticas básicas de la cartera."""
        self.mean = float(self.portfolio_returns.mean())
        self.std = float(self.portfolio_returns.std())

        '''Cálculo de retornos de la cartera'''
        return_dic = {}
        for ticker, ps in self.assets.items():
            return_dic[ticker] = ps.returns()
            
        
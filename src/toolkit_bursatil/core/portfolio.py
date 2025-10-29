"""
Clase Portfolio
---------------
Representa una cartera compuesta por varios PriceSeries y sus pesos asociados.
Calcula retornos, rentabilidad media y volatilidad conjunta.
"""

from dataclasses import dataclass, field
from src.toolkit_bursatil.core.price_series import PriceSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolkit_bursatil.core.montecarlo import MonteCarloSimulacion



@dataclass
class Portfolio:
    name: str
    assets: dict[str, PriceSeries]
    weights: dict[str, float]
    mean: float = field(init=False)
    std: float = field(init=False)
    returns_df : pd.DataFrame = field(init=False)
    portfolio_returns: pd.Series = field(init=False) 


    def __post_init__(self):
        '''Validar que los pesos suman 1'''
        if not sum(self.weights.values()) == 1.00:
            raise ValueError("Los pesos deben sumar 1")

        '''Cálculo de retornos de la cartera'''
        return_dic = {}
        for ticker, ps in self.assets.items():
            return_dic[ticker] = ps.returns()
        self.returns_df = pd.DataFrame(return_dic).dropna()

        '''Cálculo del retorno de la cartera con los pesos de los activos
        Producto entre la matriz de retornos y el vector de pesos'''
        serie_pesos = pd.Series(self.weights)
        self.portfolio_returns = self.returns_df.dot(serie_pesos)

        """Cálculo de estadísticas básicas de la cartera."""
        self.mean = float(self.portfolio_returns.mean())
        self.std = float(self.portfolio_returns.std())

    def plot_returns(self):
        """Grafica los retornos de la cartera."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.portfolio_returns.index, (1 + self.portfolio_returns).cumprod(), label='Cumulative Returns')
        plt.title(f'Cumulative Returns of Portfolio: {self.name}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.show()
    
    def simular_montecarlo(
        self,
        n_sim: int = 1000,
        horizonte: int = 252,
        valor_inicial: float = 100.0,
        tipo_retornos: str = "log",
        seed: int | None = None,
    ):
        """Ejecuta una simulación Monte Carlo sobre la cartera."""
        sim = MonteCarloSimulacion(
            objeto=self,
            n_sim=n_sim,
            horizonte=horizonte,
            valor_inicial=valor_inicial,
            tipo_retornos=tipo_retornos,
            seed=seed,
        )
        resultados = sim.ejecutar()
        self.simulacion = sim
        return resultados


    def mostrar_simulacion(self, n: int = 50):
        """Muestra visualmente los resultados de la última simulación."""
        if not hasattr(self, "simulacion"):
            raise ValueError("Primero debes ejecutar .simular_montecarlo()")
        self.simulacion.graficar(n)
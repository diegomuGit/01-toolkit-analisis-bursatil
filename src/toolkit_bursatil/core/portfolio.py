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

    # ... (Clase Portfolio y otros métodos) ...

    def plot_returns(self, tipo: str = "portfolio"):
        """
        Grafica los retornos acumulados de la cartera o de los activos individuales.

        tipo='portfolio' (defecto) → Grafica el retorno acumulado de la cartera.
        tipo='assets' → Grafica el retorno acumulado de cada activo por separado.
        """
        plt.figure(figsize=(12, 6))

        if tipo == 'portfolio':
            # Graficar solo el retorno de la cartera
            cum_returns = (1 + self.portfolio_returns).cumprod()
            cum_returns.plot(label=f'Cartera: {self.name}', legend=True)
            title = f'Rentabilidad Acumulada del Portafolio: {self.name}'

        elif tipo == 'assets':
            # Graficar los retornos individuales de cada activo
            # Usamos returns_df, rellenamos el primer NaN con 0 para el cumprod
            cum_returns_assets = (1 + self.returns_df).cumprod()
            cum_returns_assets.plot()
            title = f'Rentabilidad Acumulada de Activos Individuales en {self.name}'
        
        else:
            raise ValueError("El parámetro 'tipo' debe ser 'portfolio' o 'assets'.")

        plt.title(title)
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado')
        plt.grid(True)
        plt.show()
    
    def simular_montecarlo(
        self,
        n_sim: int = 1000,
        horizonte: int = 252,
        valor_inicial: float = 100.0,
        tipo_retornos: str = "log",
        seed: int | None = None,
    ):
        from toolkit_bursatil.core.montecarlo import MonteCarloSimulacion

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


    def resumen_simulacion(self):
            """Muestra el resumen de la simulación Monte Carlo de la serie."""
            if not hasattr(self, "simulacion"):
                # 'self.simulacion' es la instancia de MonteCarloSimulacion guardada
                raise ValueError("Primero debes ejecutar .simular_montecarlo()")
            self.simulacion.resumen() 

    def mostrar_simulacion(self, n: int = 50):
        """Muestra visualmente los resultados de la última simulación."""
        if not hasattr(self, "simulacion"):
            raise ValueError("Primero debes ejecutar .simular_montecarlo()")
        self.simulacion.graficar(n)

    def mostrar_simulacion_seaborn(self, n: int = 50):
        """Muestra visualmente los resultados de la última simulación."""
        if not hasattr(self, "simulacion"):
            raise ValueError("Primero debes ejecutar .simular_montecarlo()")
        self.simulacion.graficar_seaborn(n)
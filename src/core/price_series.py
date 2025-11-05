"""
Clase PriceSeries
-----------------
Representa una serie temporal de precios de un activo financiero.

✔️ Calcula automáticamente estadísticas básicas (media, desviación típica)
✔️ Limpia y estandariza columnas (por ejemplo, MultiIndex de yfinance) en base a la API de precios
✔️ Permite obtener retornos simples o logarítmicos
✔️ Prepara el terreno para integrarse en clases Portfolio o DataProvider
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PriceSeries:
    ticker: str
    data: pd.DataFrame
    mean: float = field(init=False)
    std: float = field(init=False)


    def __post_init__(self):
        """Limpieza y cálculo de estadísticas básicas."""
        self.data = self._prepare_data(self.data)
        returns = self.data["close"].pct_change().dropna()
        self.mean = float(returns.mean())
        self.std = float(returns.std())

    # ============================================================
    # Métodos privados (internos)
    # ============================================================

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verifica que los datos tengan las columnas necesarias y estén ordenados."""
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        if "close" not in df.columns:
            raise ValueError(f"Falta la columna 'close'.")
        return df.sort_index()



    # ============================================================
    # Métodos públicos
    # ============================================================

    def returns(self, method: str = "simple") -> pd.Series:
        """
        Devuelve la serie de retornos diarios.
        method='simple' → (P_t / P_{t-1}) - 1
        method='log' → ln(P_t / P_{t-1})
        """
        close = self.data["close"]
        if method == "log":
            return close.apply(np.log).diff().dropna()
        return close.pct_change().dropna()

    def plots_report(self, column: str = "close", title: str | None = None):
        """Grafica la serie de precios o la columna especificada."""
        if column not in self.data.columns:
            raise ValueError(f"La columna '{column}' no existe en los datos.")
        title = title or f"{self.ticker} - {column.capitalize()}"
        self.data[column].plot(title=title, figsize=(10, 4))
        plt.grid(True)
        plt.show()

    def info(self):
        """Muestra resumen de información de la serie."""
        print("═════════════════════════════════════════════")
        print(f" Ticker: {self.ticker}")
        print(
            f" Fechas: {self.data.index.min().date()} → {self.data.index.max().date()}"
        )
        print(f" Media diaria: {self.mean:.6f}")
        print(f" Desv. típica diaria: {self.std:.6f}")
        print("═════════════════════════════════════════════")

    def simular_montecarlo(
        self,
        n_sim: int = 1000,
        horizonte: int = 252,
        valor_inicial: float | None = None,
        tipo_retornos: str = "log",
        seed: int | None = None,
    ):
        """Ejecuta una simulación Monte Carlo sobre esta serie de precios."""
        if valor_inicial is None:
            valor_inicial = float(self.data["close"].iloc[-1])

        from src.core.montecarlo import MonteCarloSimulacion

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
        """Muestra el gráfico de la simulación Monte Carlo de la serie."""
        if not hasattr(self, "simulacion"):
            raise ValueError("Primero debes ejecutar .simular_montecarlo()")
        self.simulacion.graficar(n)
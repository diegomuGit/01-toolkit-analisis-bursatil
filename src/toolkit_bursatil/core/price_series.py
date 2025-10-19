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
        """Estandariza columnas: aplanar MultiIndex, pasar a minúsculas y ordenar."""
    
        df = df.copy()

        # Si las columnas son MultiIndex (como en yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            level_names = list(df.columns.names)

            # Caso 1: niveles ('Price', 'Ticker') -> queremos el de precios
            if "Price" in level_names and "Ticker" in level_names:
                df.columns = df.columns.get_level_values("Price")

            # Caso 2: niveles ('Ticker', 'Price') -> también lo manejamos
            elif "Ticker" in level_names and "Price" in level_names[::-1]:
                df.columns = df.columns.get_level_values("Price")

            # Si no tiene nombres claros, cogemos siempre el último nivel
            else:
                df.columns = df.columns.get_level_values(-1)

        # Normalizar nombres
        df.columns = [c.lower().strip() for c in df.columns]

        # Validar que existe la columna 'close'
        if "close" not in df.columns:
            raise ValueError(
                f"El DataFrame no contiene columna 'close'. Columnas detectadas: {df.columns.tolist()}"
            )

        # Ordenar por fecha (por si viene descendente)
        df = df.sort_index()
        return df


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

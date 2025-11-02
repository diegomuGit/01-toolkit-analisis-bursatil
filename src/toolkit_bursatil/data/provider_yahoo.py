from dataclasses import dataclass
from toolkit_bursatil.core.price_series import PriceSeries
import yfinance as yf
import pandas as pd
from toolkit_bursatil.data.provider_base import DataProviderBase


@dataclass
class YahooSerie(DataProviderBase):
    ticker: str
    start_date: str
    end_date: str

    def get_serie_precios(self) -> PriceSeries:
        """Descarga la serie histórica de precios desde Yahoo Finance."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)

        # --- Normalización mínima para evitar el error de 'close' ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values("Price")  # nos quedamos con el nivel de campos

        data.columns = [c.lower().strip() for c in data.columns]

        '''# Renombrar 'adj close' a 'close' si 'close' no existe
        if "close" not in data.columns and "adj close" in data.columns:
            data = data.rename(columns={"adj close": "close"})'''

        if "close" not in data.columns:
            raise ValueError(
                f"YahooSerie: el DataFrame no contiene columna 'close'. Columnas detectadas: {data.columns.tolist()}"
            )

        data = data.sort_index()

        return PriceSeries(ticker=self.ticker, data=data)


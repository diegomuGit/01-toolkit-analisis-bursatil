from dataclasses import dataclass
from typing import Optional
from src.core.price_series import PriceSeries
from src.data.provider_base import DataProviderBase
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import pandas as pd
import os

class AlphaVantageSerie(DataProviderBase):
    """Descarga series de precios desde Alpha Vantage y devuelve un objeto PriceSeries."""
    def get_serie_precios(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> PriceSeries:
        """Descarga la serie histórica de precios desde Alpha Vantage."""
        load_dotenv()
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")

        if not api_key:
            raise ValueError("❌ No se encontró la clave 'ALPHAVANTAGE_API_KEY' en el archivo .env")

        ts = TimeSeries(key=api_key, output_format="pandas")
        data, *resto = ts.get_daily(symbol=ticker, outputsize="full") # datos diarios. No estoy usando precios ajustados. Para usar esos datos de esta api, desde la funcion get_daily_adjusted se obtienen los precios ajustados pagando

        # Renombrar columnas al formato estándar
        data = data.rename( # type: ignore
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )

        data.columns = [c.lower().strip() for c in data.columns]
        print("Columnas estandarizadas:", data.columns)
        data = data.sort_index()

        # Recortar rango de fechas
        data = data.loc[start:end]

        if "close" not in data.columns:
            raise ValueError(f"❌ Falta la columna 'close'. Columnas detectadas: {data.columns.tolist()}")

        return PriceSeries(ticker=ticker, data=data)

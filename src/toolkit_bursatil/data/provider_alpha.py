from dataclasses import dataclass
from toolkit_bursatil.core.price_series import PriceSeries
from toolkit_bursatil.data.provider_base import DataProviderBase
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import pandas as pd
import os


@dataclass
class AlphaVantageSerie(DataProviderBase):
    """Descarga series de precios desde Alpha Vantage y devuelve un objeto PriceSeries."""

    ticker: str
    start_date: str
    end_date: str

    def get_serie_precios(self) -> PriceSeries:
        """Descarga la serie histórica de precios desde Alpha Vantage."""
        load_dotenv()
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")

        if not api_key:
            raise ValueError("❌ No se encontró la clave 'ALPHAVANTAGE_API_KEY' en el archivo .env")

        ts = TimeSeries(key=api_key, output_format="pandas")
        data, *resto = ts.get_daily(symbol=self.ticker, outputsize="full") # datos diarios

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
        data = data.loc[self.start_date:self.end_date]

        if "close" not in data.columns:
            raise ValueError(f"❌ Falta la columna 'close'. Columnas detectadas: {data.columns.tolist()}")

        return PriceSeries(ticker=self.ticker, data=data)

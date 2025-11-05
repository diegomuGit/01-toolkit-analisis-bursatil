"""
Clase YahooFundamentals
-----------------------
Obtiene información fundamental de empresas desde Yahoo Finance.

✔️ Balance Sheet (Estado de Situación Financiera)
✔️ Income Statement (Estado de Resultados)
✔️ Cash Flow (Flujo de Efectivo)
✔️ Información de la empresa
"""

import yfinance as yf
import pandas as pd
from typing import Optional


class YahooFundamentals:
    """Descarga datos fundamentales desde Yahoo Finance."""
    
    def __init__(self, ticker: str):
        """
        Inicializa el proveedor de datos fundamentales.
        
        Args:
            ticker: Símbolo del activo (ej: 'AAPL', 'MSFT')
        """
        self.ticker = ticker
        self._ticker_obj = None
    
    def _get_ticker(self):
        """Obtiene o reutiliza el objeto Ticker de yfinance."""
        if self._ticker_obj is None:
            self._ticker_obj = yf.Ticker(self.ticker)
        return self._ticker_obj
    
    def get_balance_sheet(self, quarterly: bool = False) -> pd.DataFrame:
        """
        Obtiene el Balance General (Estado de Situación Financiera).
        
        Args:
            quarterly: Si True, obtiene datos trimestrales. 
                      Si False, obtiene datos anuales.
        
        Returns:
            DataFrame con el balance sheet. Las columnas son fechas
            y las filas son diferentes líneas del balance (activos, 
            pasivos, patrimonio, etc.)
        
        Raises:
            ValueError: Si no se pueden obtener datos para el ticker
        
        Example:
            >>> fundamentals = YahooFundamentals("AAPL")
            >>> balance = fundamentals.get_balance_sheet()
            >>> print(balance.loc["Total Assets"])
        """
        ticker_obj = self._get_ticker()
        
        try:
            if quarterly:
                data = ticker_obj.quarterly_balance_sheet
            else:
                data = ticker_obj.balance_sheet
            
            if data is None or data.empty:
                raise ValueError(
                    f"No se obtuvieron datos de balance sheet para '{self.ticker}'. "
                    f"Verifica que el ticker sea válido."
                )
            
            return data
            
        except Exception as e:
            raise ValueError(
                f"Error al obtener balance sheet para '{self.ticker}': {str(e)}"
            )
    
    def get_income_statement(self, quarterly: bool = False) -> pd.DataFrame:
        """
        Obtiene el Estado de Resultados.
        
        Args:
            quarterly: Si True, obtiene datos trimestrales.
        
        Returns:
            DataFrame con ingresos, costos, utilidades, etc.
        """
        ticker_obj = self._get_ticker()
        
        try:
            if quarterly:
                data = ticker_obj.quarterly_financials
            else:
                data = ticker_obj.financials
            
            if data is None or data.empty:
                raise ValueError(
                    f"No se obtuvieron datos de income statement para '{self.ticker}'"
                )
            
            return data
            
        except Exception as e:
            raise ValueError(
                f"Error al obtener income statement para '{self.ticker}': {str(e)}"
            )
    
    def get_cash_flow(self, quarterly: bool = False) -> pd.DataFrame:
        """
        Obtiene el Estado de Flujo de Efectivo.
        
        Args:
            quarterly: Si True, obtiene datos trimestrales.
        
        Returns:
            DataFrame con flujos de efectivo operativos, de inversión 
            y financiamiento.
        """
        ticker_obj = self._get_ticker()
        
        try:
            if quarterly:
                data = ticker_obj.quarterly_cashflow
            else:
                data = ticker_obj.cashflow
            
            if data is None or data.empty:
                raise ValueError(
                    f"No se obtuvieron datos de cash flow para '{self.ticker}'"
                )
            
            return data
            
        except Exception as e:
            raise ValueError(
                f"Error al obtener cash flow para '{self.ticker}': {str(e)}"
            )
    
    def get_info(self) -> dict:
        """
        Obtiene información general de la empresa.
        
        Returns:
            Diccionario con datos como sector, industria, market cap,
            P/E ratio, dividend yield, etc.
        """
        ticker_obj = self._get_ticker()
        return ticker_obj.info
    
    def resumen(self):
        """Muestra un resumen de la información disponible."""
        info = self.get_info()
        
        print(f"\n{'='*60}")
        print(f"RESUMEN FUNDAMENTAL - {self.ticker}")
        print(f"{'='*60}\n")
        
        print(f"Nombre: {info.get('longName', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industria: {info.get('industry', 'N/A')}")
        print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "Dividend Yield: N/A")
        
        print(f"\n{'='*60}\n")
"""
Clase Portfolio
---------------
Representa una cartera compuesta por varios PriceSeries y sus pesos asociados.
Calcula retornos, rentabilidad media y volatilidad conjunta.
"""

from dataclasses import dataclass, field
from toolkit_bursatil.core.price_series import PriceSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



@dataclass
class Portfolio:
    name: str
    assets: dict[str, PriceSeries]
    weights: dict[str, float]
    mean: float = field(init=False)
    std: float = field(init=False)
    returns_df : pd.DataFrame = field(init=False)
    portfolio_returns: pd.Series = field(init=False)
    cov_matrix: pd.DataFrame = field(init=False)
    corr_matrix: pd.DataFrame = field(init=False) 


    def __post_init__(self):
        '''Validar que los pesos suman 1'''
        if not sum(self.weights.values()) == 1.00:
            raise ValueError("Los pesos deben sumar 1")

        '''Cálculo de retornos de la cartera'''
        return_dic = {}
        for ticker, ps in self.assets.items():
            return_dic[ticker] = ps.returns()
        self.returns_df = pd.DataFrame(return_dic).dropna()

        '''Cálculo de las matrices de covarianza y correlación'''
        self.cov_matrix = self.returns_df.cov()
        self.corr_matrix = self.returns_df.corr()

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
    
    def visualizar_covarianza(self, show_values: bool = True, cmap: str = "RdYlGn"):
        """
        Visualiza la matriz de covarianzas de la cartera.
        
        Parámetros:
        -----------
        show_values : bool, default=True
            Si True, muestra los valores numéricos en cada celda del heatmap
        cmap : str, default="RdYlGn"
            Mapa de colores para el heatmap
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            self.cov_matrix,
            annot=show_values,
            fmt='.6f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Covarianza'},
            ax=ax
        )
        
        ax.set_title(
            f'Matriz de Covarianzas - Portfolio: {self.name}',
            fontsize=14,
            weight='bold',
            pad=20
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def visualizar_correlacion(self, show_values: bool = True, cmap: str = "coolwarm"):
        """
        Visualiza la matriz de correlación de la cartera.
        
        Parámetros:
        -----------
        show_values : bool, default=True
            Si True, muestra los valores numéricos en cada celda
        cmap : str, default="coolwarm"
            Mapa de colores para el heatmap
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            self.corr_matrix,
            annot=show_values,
            fmt='.3f',
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlación'},
            ax=ax
        )
        
        ax.set_title(
            f'Matriz de Correlación - Portfolio: {self.name}',
            fontsize=14,
            weight='bold',
            pad=20
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def report(
        self,
        mostrar: bool = True,
        incluir_var: bool = True,
        nivel_confianza: float = 0.95,
        periodo_anual: int = 252,
    ) -> str:
        """
        Genera un informe simplificado y austero de la cartera en formato Markdown.
        
        Este informe se centra en el rendimiento agregado de la cartera,
        la composición de pesos y el Value at Risk (VaR).
        
        Parámetros:
        -----------
        mostrar : bool, default=True
            Si es True, imprime el reporte en pantalla.
        incluir_var : bool, default=True
            Incluye cálculo de Value at Risk (VaR) y CVaR.
        nivel_confianza : float, default=0.95
            Nivel de confianza para el cálculo del VaR.
        periodo_anual : int, default=252
            Número de periodos para anualización (252 días de trading).
        
        Returns:
        --------
        str : Reporte en formato Markdown
        """
        lines = []
        
        # ===== ENCABEZADO =====
        lines.append(f"# 📊 Informe de Cartera: {self.name}")
        lines.append("")
        lines.append(f"**Fecha de generación:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # ===== RESUMEN EJECUTIVO =====
        lines.append("## 📈 Resumen Ejecutivo")
        lines.append("")
        
        # Métricas principales
        retorno_anualizado = self.mean * periodo_anual
        volatilidad_anualizada = self.std * np.sqrt(periodo_anual)
        sharpe_ratio = retorno_anualizado / volatilidad_anualizada if volatilidad_anualizada > 0 else 0
        
        lines.append("| Métrica                    | Valor           |")
        lines.append("|:---------------------------|----------------:|")
        lines.append(f"| Retorno Promedio Diario    | {self.mean*100:>7.4f}%  |")
        lines.append(f"| Retorno Anualizado         | {retorno_anualizado*100:>7.2f}%  |")
        lines.append(f"| Volatilidad Diaria         | {self.std*100:>7.4f}%  |")
        lines.append(f"| Volatilidad Anualizada     | {volatilidad_anualizada*100:>7.2f}%  |")
        lines.append(f"| Ratio de Sharpe            | {sharpe_ratio:>7.4f}   |")
        lines.append(f"| Número de Activos          | {len(self.assets):>7d}   |")
        lines.append(f"| Observaciones              | {len(self.portfolio_returns):>7d}   |")
        lines.append("")
        
        # ===== COMPOSICIÓN DE LA CARTERA =====
        lines.append("## 🎯 Composición de la Cartera")
        lines.append("")
        lines.append("| Ticker | Peso    | Contribución Diaria |")
        lines.append("|:-------|--------:|--------------------:|")
        
        for ticker, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            contribucion = self.returns_df[ticker].mean() * weight * 100
            lines.append(f"| {ticker:<6} | {weight*100:>6.2f}% | {contribucion:>10.4f}%      |")
        
        lines.append("")
        
        # ===== VALUE AT RISK (VaR) =====
        if incluir_var:
            lines.append("## ⚠️ Value at Risk (VaR)")
            lines.append("")
            
            var_percentil = (1 - nivel_confianza) * 100
            var_historico = np.percentile(self.portfolio_returns, var_percentil)
            cvar = self.portfolio_returns[self.portfolio_returns <= var_historico].mean()
            
            lines.append(f"**Nivel de confianza:** {nivel_confianza*100:.0f}%")
            lines.append("")
            lines.append("| Métrica                   | Diario      | Anualizado  |")
            lines.append("|:--------------------------|------------:|------------:|")
            lines.append(f"| VaR ({nivel_confianza*100:.0f}%)              | {var_historico*100:>10.4f}% | {var_historico*np.sqrt(periodo_anual)*100:>10.2f}% |")
            lines.append(f"| CVaR (Expected Shortfall) | {cvar*100:>10.4f}% | {cvar*np.sqrt(periodo_anual)*100:>10.2f}% |")
            lines.append("")
        
        # ===== ESTADÍSTICAS ADICIONALES =====
        lines.append("## 📉 Estadísticas de Distribución")
        lines.append("")
        
        skewness = self.portfolio_returns.skew()
        kurtosis = self.portfolio_returns.kurtosis()
        retorno_acumulado = ((1 + self.portfolio_returns).prod() - 1) * 100
        
        # Interpretaciones
        interp_skew = (
            "Sesgo negativo (más pérdidas extremas)" if skewness < -0.5 
            else "Sesgo positivo (más ganancias extremas)" if skewness > 0.5 
            else "Distribución simétrica"
        )
        
        interp_kurt = (
            "Colas pesadas (mayor riesgo extremo)" if kurtosis > 1 
            else "Colas ligeras" if kurtosis < -1 
            else "Similar a distribución normal"
        )
        
        lines.append("| Métrica                  | Valor      | Interpretación                          |")
        lines.append("|:-------------------------|:----------:|:----------------------------------------|")
        lines.append(f"| Asimetría (Skewness)     | {skewness:>9.4f} | {interp_skew:<40} |")
        lines.append(f"| Curtosis (Kurtosis)      | {kurtosis:>9.4f} | {interp_kurt:<40} |")
        lines.append(f"| Retorno Acumulado Total  | {retorno_acumulado:>8.2f}% | Retorno total del periodo               |")
        lines.append("")
        
        markdown_report = "\n".join(lines)
        
        if mostrar:
            print(markdown_report)
        
        return markdown_report
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
    
    def report(
        self,
        mostrar: bool = True,
        incluir_activos: bool = True,
        incluir_correlacion: bool = True,
        incluir_var: bool = True,
        nivel_confianza: float = 0.95,
        periodo_anual: int = 252,
    ) -> str:
        """
        Genera un informe completo de la cartera en formato Markdown.
        
        Parámetros:
        -----------
        mostrar : bool, default=True
            Si es True, imprime el reporte en pantalla.
        incluir_activos : bool, default=True
            Incluye análisis individual de cada activo.
        incluir_correlacion : bool, default=True
            Incluye matriz de correlación entre activos.
        incluir_var : bool, default=True
            Incluye cálculo de Value at Risk (VaR).
        nivel_confianza : float, default=0.95
            Nivel de confianza para el cálculo del VaR.
        periodo_anual : int, default=252
            Número de periodos para anualización (252 días de trading).
            
        Returns:
        --------
        str : Reporte en formato Markdown
        """
        
        # Construcción del reporte
        lines = []
        lines.append(f"# 📊 Informe de Cartera: {self.name}")
        lines.append("")
        lines.append(f"**Fecha de generación:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # ===== RESUMEN EJECUTIVO =====
        lines.append("## 📈 Resumen Ejecutivo")
        lines.append("")
        
        # Métricas principales
        retorno_anualizado = self.mean * periodo_anual
        volatilidad_anualizada = self.std * np.sqrt(periodo_anual)
        sharpe_ratio = retorno_anualizado / volatilidad_anualizada if volatilidad_anualizada > 0 else 0
        
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| **Retorno Promedio Diario** | {self.mean*100:.4f}% |")
        lines.append(f"| **Retorno Anualizado** | {retorno_anualizado*100:.2f}% |")
        lines.append(f"| **Volatilidad Diaria** | {self.std*100:.4f}% |")
        lines.append(f"| **Volatilidad Anualizada** | {volatilidad_anualizada*100:.2f}% |")
        lines.append(f"| **Ratio de Sharpe** | {sharpe_ratio:.4f} |")
        lines.append(f"| **Número de Activos** | {len(self.assets)} |")
        lines.append(f"| **Observaciones** | {len(self.portfolio_returns)} |")
        lines.append("")
        
        # ===== COMPOSICIÓN DE LA CARTERA =====
        lines.append("## 💼 Composición de la Cartera")
        lines.append("")
        lines.append("| Ticker | Peso | Contribución al Retorno |")
        lines.append("|--------|------|-------------------------|")
        
        for ticker, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            contribucion = self.returns_df[ticker].mean() * weight * 100
            lines.append(f"| {ticker} | {weight*100:.2f}% | {contribucion:.4f}% |")
        lines.append("")
        
        # ===== ADVERTENCIAS Y ALERTAS =====
        lines.append("## ⚠️ Advertencias y Alertas")
        lines.append("")
        
        advertencias = []
        
        # Advertencia: Concentración
        max_peso = max(self.weights.values())
        if max_peso > 0.40:
            advertencias.append(f"🔴 **Alta concentración:** Un activo representa más del 40% de la cartera ({max_peso*100:.1f}%)")
        elif max_peso > 0.30:
            advertencias.append(f"🟡 **Concentración moderada:** Un activo representa más del 30% de la cartera ({max_peso*100:.1f}%)")
        
        # Advertencia: Volatilidad
        if volatilidad_anualizada > 0.30:
            advertencias.append(f"🔴 **Alta volatilidad:** La volatilidad anualizada es del {volatilidad_anualizada*100:.1f}%")
        elif volatilidad_anualizada > 0.20:
            advertencias.append(f"🟡 **Volatilidad moderada-alta:** La volatilidad anualizada es del {volatilidad_anualizada*100:.1f}%")
        
        # Advertencia: Sharpe Ratio
        if sharpe_ratio < 0:
            advertencias.append(f"🔴 **Sharpe Ratio negativo:** La cartera tiene un ratio de Sharpe de {sharpe_ratio:.2f}")
        elif sharpe_ratio < 0.5:
            advertencias.append(f"🟡 **Sharpe Ratio bajo:** El ratio de Sharpe es {sharpe_ratio:.2f} (recomendado > 1.0)")
        
        # Advertencia: Retorno negativo
        if retorno_anualizado < 0:
            advertencias.append(f"🔴 **Retorno negativo:** La cartera tiene un retorno anualizado de {retorno_anualizado*100:.2f}%")
        
        # Advertencia: Número de activos
        if len(self.assets) < 3:
            advertencias.append(f"🟡 **Diversificación limitada:** La cartera tiene solo {len(self.assets)} activo(s)")
        
        if advertencias:
            for adv in advertencias:
                lines.append(f"- {adv}")
        else:
            lines.append("✅ **No se detectaron advertencias significativas.**")
        lines.append("")
        
        # ===== VALUE AT RISK (VaR) =====
        if incluir_var:
            lines.append("## 📉 Value at Risk (VaR)")
            lines.append("")
            
            var_parametrico = np.percentile(self.portfolio_returns, (1 - nivel_confianza) * 100)
            cvar = self.portfolio_returns[self.portfolio_returns <= var_parametrico].mean()
            
            lines.append(f"**Nivel de confianza:** {nivel_confianza*100:.0f}%")
            lines.append("")
            lines.append("| Métrica | Diario | Anualizado |")
            lines.append("|---------|--------|------------|")
            lines.append(f"| **VaR ({nivel_confianza*100:.0f}%)** | {var_parametrico*100:.4f}% | {var_parametrico*np.sqrt(periodo_anual)*100:.2f}% |")
            lines.append(f"| **CVaR (Expected Shortfall)** | {cvar*100:.4f}% | {cvar*np.sqrt(periodo_anual)*100:.2f}% |")
            lines.append("")
            lines.append(f"*Con un {nivel_confianza*100:.0f}% de confianza, la pérdida máxima esperada en un día es de {abs(var_parametrico)*100:.2f}%*")
            lines.append("")
        
        # ===== ANÁLISIS POR ACTIVO =====
        if incluir_activos:
            lines.append("## 📊 Análisis Individual de Activos")
            lines.append("")
            
            for ticker in sorted(self.weights.keys()):
                asset_returns = self.returns_df[ticker]
                asset_mean = asset_returns.mean() * periodo_anual
                asset_std = asset_returns.std() * np.sqrt(periodo_anual)
                asset_sharpe = asset_mean / asset_std if asset_std > 0 else 0
                
                lines.append(f"### {ticker} ({self.weights[ticker]*100:.1f}% de la cartera)")
                lines.append("")
                lines.append("| Métrica | Valor |")
                lines.append("|---------|-------|")
                lines.append(f"| Retorno Anualizado | {asset_mean*100:.2f}% |")
                lines.append(f"| Volatilidad Anualizada | {asset_std*100:.2f}% |")
                lines.append(f"| Sharpe Ratio | {asset_sharpe:.4f} |")
                lines.append(f"| Mínimo | {asset_returns.min()*100:.2f}% |")
                lines.append(f"| Máximo | {asset_returns.max()*100:.2f}% |")
                lines.append("")
        
        # ===== MATRIZ DE CORRELACIÓN =====
        if incluir_correlacion and len(self.assets) > 1:
            lines.append("## 🔗 Matriz de Correlación")
            lines.append("")
            
            corr_matrix = self.returns_df.corr()
            
            # Crear tabla markdown
            header = "| " + " | ".join([""] + list(corr_matrix.columns)) + " |"
            separator = "|" + "|".join(["---"] * (len(corr_matrix.columns) + 1)) + "|"
            lines.append(header)
            lines.append(separator)
            
            for idx in corr_matrix.index:
                row_values = [f"{val:.3f}" for val in corr_matrix.loc[idx]]
                row = f"| **{idx}** | " + " | ".join(row_values) + " |"
                lines.append(row)
            lines.append("")
            
            # Análisis de correlaciones
            lines.append("**Análisis de correlaciones:**")
            lines.append("")
            
            # Buscar correlaciones altas
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr:
                lines.append("⚠️ **Correlaciones altas detectadas (|r| > 0.8):**")
                for t1, t2, corr in high_corr:
                    lines.append(f"- {t1} ↔ {t2}: {corr:.3f}")
            else:
                lines.append("✅ No se detectaron correlaciones excesivamente altas entre activos.")
            lines.append("")
        
        # ===== ESTADÍSTICAS ADICIONALES =====
        lines.append("## 📊 Estadísticas Adicionales")
        lines.append("")
        
        skewness = self.portfolio_returns.skew()
        kurtosis = self.portfolio_returns.kurtosis()
        
        lines.append("| Métrica | Valor | Interpretación |")
        lines.append("|---------|-------|----------------|")
        lines.append(f"| **Asimetría (Skewness)** | {skewness:.4f} | {'Sesgo negativo (más pérdidas extremas)' if skewness < -0.5 else 'Sesgo positivo (más ganancias extremas)' if skewness > 0.5 else 'Distribución simétrica'} |")
        lines.append(f"| **Curtosis (Kurtosis)** | {kurtosis:.4f} | {'Colas pesadas (eventos extremos)' if kurtosis > 1 else 'Colas ligeras' if kurtosis < -1 else 'Similar a normal'} |")
        lines.append(f"| **Retorno Acumulado** | {((1 + self.portfolio_returns).prod() - 1)*100:.2f}% | Total en el periodo |")
        lines.append("")
        
        # ===== PIE DE PÁGINA =====
        lines.append("---")
        lines.append("*Este informe es generado automáticamente y tiene fines informativos. No constituye asesoramiento financiero.*")
        
        # Unir todas las líneas
        markdown_report = "\n".join(lines)
        
        # Mostrar si se solicita
        if mostrar:
            try:
                from IPython.display import display, Markdown
                display(Markdown(markdown_report))
            except ImportError:
                print(markdown_report)
        
        return markdown_report
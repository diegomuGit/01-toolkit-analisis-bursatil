from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.core.price_series import PriceSeries
from src.core.portfolio import Portfolio


@dataclass
class MonteCarloSimulacion:
    """Simulación de Monte Carlo para una PriceSeries o Portfolio."""

    objeto: PriceSeries | Portfolio  # Puede ser PriceSeries o Portfolio
    n_sim: int = 1000            # Nº de simulaciones
    horizonte: int = 252         # Días a simular
    valor_inicial: float = 100.0
    tipo_retornos: str = "log"   # 'log' o 'simple'
    seed: int | None = None


    def ejecutar(self) -> pd.DataFrame:
        """
        Ejecuta la simulación de Monte Carlo, discriminando entre un objeto
        PriceSeries (univariado) y un objeto Portfolio (multivariado/Cholesky).
        """
        np.random.seed(self.seed)
        
        # 1. DISCRIMINACIÓN DEL OBJETO Y EXTRACCIÓN DE PARÁMETROS
        
        # Caso A: Objeto es PriceSeries (Simulación Univariada Original)
        if isinstance(self.objeto, PriceSeries):
            
            # CORREGIDO: Obtenemos los retornos históricos correctos (log o simple)
            #           en lugar de usar los que venían por defecto en PriceSeries.
            if self.tipo_retornos == "log":
                returns_hist = self.objeto.returns(method="log")
            else:
                returns_hist = self.objeto.returns(method="simple")
                
            mu = returns_hist.mean()
            sigma = returns_hist.std()
            
            # Generar retornos simulados (basados en distribución normal simple)
            retornos_simulados = np.random.normal(mu, sigma, (self.horizonte, self.n_sim))
            retornos_simulados_df = pd.DataFrame(retornos_simulados)
            
            # El flujo sigue el cálculo univariado
            if self.tipo_retornos == "log":
                # Compounding logarítmico
                precios = self.valor_inicial * np.exp(retornos_simulados_df.cumsum())
            else:
                # Compounding simple
                precios = self.valor_inicial * (1 + retornos_simulados_df).cumprod()
            
            # Ensure precios is a DataFrame
            if not isinstance(precios, pd.DataFrame):
                precios = pd.DataFrame(precios)
                
            self.resultados = precios
            return precios

        # Caso B: Objeto es Portfolio (Simulación Multivariada - Cholesky)
        elif isinstance(self.objeto, Portfolio):
            
            # --- PASO 1: Obtener datos de la cartera (CORREGIDO) ---
            # Extraemos los retornos del tipo correcto (log o simple)
            
            if self.tipo_retornos == "log":
                # NUEVO: Si es log, recalculamos los retornos de los activos como log
                return_dic = {
                    ticker: ps.returns(method="log") 
                    for ticker, ps in self.objeto.assets.items()
                }
                returns_df = pd.DataFrame(return_dic).dropna()
            else:
                # Si es simple, usamos el que ya calculó el Portfolio
                returns_df = self.objeto.returns_df

            mu_vector = returns_df.mean()    # Vector de medias (drift)
            cov_matrix = returns_df.cov()   # Matriz de Covarianza desde Portfolio

            n_assets = len(mu_vector)
            pesos = pd.Series(self.objeto.weights)

            # --- PASO 2: Descomposición de Cholesky ---
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("La matriz de covarianza no es semi-definida positiva. Revisa tus datos.")

            # --- PASO 3, 4 y 5: Generar Retornos (CORREGIDO) ---
            # El bucle de simulación AHORA contiene la generación de números aleatorios.
            
            simulaciones_finales = {}

            for i in range(self.n_sim):
                
                # 1. Generar aleatorios (Z) para ESTA simulación (i)
                Z = np.random.normal(size=(self.horizonte, n_assets))
                
                # 2. Correlacionar y añadir drift
                # Epsilon = Z @ L.T
                retornos_correlacionados = Z @ L.T
                # Sumamos el drift (media de retornos log o simples, según corresponda)
                retornos_simulados_con_drift = retornos_correlacionados + np.asarray(mu_vector.values)

                # 3. Convertir a retornos simples (si partimos de log)
                if self.tipo_retornos == "log":
                    # Si simulamos log-returns, los convertimos a simples
                    retornos_simples_simulados = np.exp(retornos_simulados_con_drift) - 1
                    retornos_simples_df = pd.DataFrame(retornos_simples_simulados, columns=mu_vector.index)
                else:
                    # Si simulamos simples, ya los tenemos
                    retornos_simples_df = pd.DataFrame(retornos_simulados_con_drift, columns=mu_vector.index)
                
                # 4. Calcular el retorno diario de la cartera para esta simulación
                portfolio_daily_returns_i = retornos_simples_df.dot(pesos)
                
                # 5. Calcular la trayectoria de valor final
                valor_acumulado = (1 + portfolio_daily_returns_i).cumprod()
                simulaciones_finales[f'Simulacion_{i+1}'] = self.valor_inicial * valor_acumulado

            # Convertir el diccionario de trayectorias en el DataFrame de resultados
            precios = pd.DataFrame(simulaciones_finales)

            self.resultados = precios # self.resultados ahora es un DataFrame con n_sim columnas
            return precios

        else:
            # Manejo de error si se pasa un tipo de objeto no compatible
            raise TypeError("El objeto para la simulación debe ser PriceSeries o Portfolio.")

    def resumen(self):
        """Muestra estadísticas básicas del resultado final."""
        finales = self.resultados.iloc[-1]
        print("═════════════════════════════════════════════")
        print("  Resultados Simulación Monte Carlo")
        print(f" Media final: {finales.mean():.2f}")
        print(f" P5: {np.percentile(finales, 5):.2f}")
        print(f" P95: {np.percentile(finales, 95):.2f}")
        print("═════════════════════════════════════════════")


    def graficar(self, n: int = 50):
        """Muestra el gráfico de N trayectorias simuladas."""
        if self.resultados is None:
            raise ValueError("Primero debes ejecutar .ejecutar()")

        # 1. Seleccionar 'n' trayectorias aleatorias (o todas si n > n_sim)
        n_sim = self.resultados.shape[1]
        n_a_mostrar = min(n, n_sim)
        
        # Selecciona N columnas al azar del DataFrame de resultados
        caminos_a_mostrar = self.resultados.sample(n=n_a_mostrar, axis=1)

        # 2. Configurar y dibujar el gráfico
        plt.figure(figsize=(12, 6))
        
        # PLOTEO: Pandas automáticamente plotea cada columna como una línea
        # Determinar el título según el tipo de objeto
        if isinstance(self.objeto, PriceSeries):
            titulo = f"Simulación Monte Carlo - {self.objeto.ticker} ({n_a_mostrar} caminos)"
        else:
            titulo = f"Simulación Monte Carlo - {self.objeto.name} ({n_a_mostrar} caminos)"
        
        caminos_a_mostrar.plot(
            ax=plt.gca(),
            legend=False,  # Ocultamos la leyenda si son muchas líneas
            alpha=0.3,     # Usamos transparencia para que se vea la densidad
            grid=True,
            title=titulo
        )
        
        plt.xlabel("Días")
        plt.ylabel("Valor simulado")
        plt.show()


    def graficar_seaborn(self, n: int = 50):
        """
        Visualiza trayectorias Monte Carlo (izquierda) y distribución de retornos logarítmicos (derecha).
        Muestra la media de los caminos y la curva teórica normal sobre los retornos.
        """
        if self.resultados is None:
            raise ValueError("Primero debes ejecutar .ejecutar()")

        # --- Selección de caminos ---
        n_sim = self.resultados.shape[1]
        n_a_mostrar = min(n, n_sim)
        caminos = self.resultados.sample(n=n_a_mostrar, axis=1)

        # --- Crear figura con 2 subplots (paths + distribución de retornos) ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
        plt.suptitle("Simulación Monte Carlo - Trayectorias y Distribución de Retornos", fontsize=14, weight="bold")

        # ====== 1️⃣ Panel Izquierdo: Paths ======
        ax1 = axes[0]
        sns.lineplot(data=caminos, ax=ax1, alpha=0.25, linewidth=1, legend=False)

        # Calcular y trazar la media en magenta
        mean_path = caminos.mean(axis=1)
        sns.lineplot(x=mean_path.index, y=mean_path.values, ax=ax1, color="magenta", linewidth=2.5, label="Media")

        ax1.set_title("Trayectorias simuladas", fontsize=12)
        ax1.set_xlabel("Días")
        ax1.set_ylabel("Valor simulado")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # ====== 2️⃣ Panel Derecho: Distribución de retornos ======
        ax2 = axes[1]
        valores_finales = caminos.iloc[-1, :]
        s0 = caminos.iloc[0, 0]

        # Retornos logarítmicos finales
        retornos_log = np.log(valores_finales / float(s0))

        # Histograma con KDE
        sns.histplot(retornos_log, ax=ax2, kde=True, color="limegreen", stat="density", bins=20, edgecolor="black", alpha=0.6)

        # Curva normal teórica sobre el histograma
        from scipy.stats import norm
        mu, sigma = retornos_log.mean(), retornos_log.std()
        x = np.linspace(retornos_log.min(), retornos_log.max(), 200)
        ax2.plot(x, norm.pdf(x, mu, sigma), color="magenta", linewidth=2, label="Normal teórica")

        ax2.set_title("Distribución de retornos logarítmicos finales", fontsize=12)
        ax2.set_xlabel("Retorno logarítmico")
        ax2.set_ylabel("Densidad")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


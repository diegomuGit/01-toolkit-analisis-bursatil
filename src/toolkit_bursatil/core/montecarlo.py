from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class MonteCarloSimulacion:
    """Simulación de Monte Carlo para una PriceSeries o Portfolio."""

    objeto: object                  # Puede ser PriceSeries o Portfolio
    n_sim: int = 1000            # Nº de simulaciones
    horizonte: int = 252         # Días a simular
    valor_inicial: float = 100.0
    tipo_retornos: str = "log"   # 'log' o 'simple'
    seed: int | None = None

    def ejecutar(self) -> pd.DataFrame:
        """Ejecuta la simulación y devuelve un DataFrame con las trayectorias simuladas."""
        np.random.seed(self.seed)

        # Determinar media y desviación según tipo de objeto
        if hasattr(self.objeto, "mean") and hasattr(self.objeto, "std"):
            mu, sigma = self.objeto.mean, self.objeto.std
        else:
            raise ValueError("El objeto debe tener atributos 'mean' y 'std'.")

        # Generar retornos simulados
        retornos = np.random.normal(mu, sigma, (self.horizonte, self.n_sim))
        retornos = pd.DataFrame(retornos)

        # Calcular precios simulados
        if self.tipo_retornos == "log":
            precios = self.valor_inicial * np.exp(retornos.cumsum())
        else:
            precios = self.valor_inicial * (1 + retornos).cumprod()

        self.resultados = precios
        return precios

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
        """Grafica n trayectorias simuladas."""
        resultados_df = pd.DataFrame(self.resultados) if not isinstance(self.resultados, pd.DataFrame) else self.resultados
        resultados_df.iloc[:, :n].plot(legend=False, alpha=0.4, figsize=(10, 4))
        titulo = getattr(self.objeto, "ticker", getattr(self.objeto, "name", "Simulación"))
        plt.title(f"Simulación Monte Carlo - {titulo}")
        plt.xlabel("Días")
        plt.ylabel("Valor simulado")
        plt.grid(True)
        plt.show()

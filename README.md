# 📊 Toolkit de Análisis Bursátil

Un proyecto didáctico en **Python** para aprender y practicar análisis de datos financieros.  
Combina **notebooks interactivos** con una **librería modular propia**, pensada para explorar, analizar y visualizar el comportamiento de los mercados bursátiles.

---

## 🎯 Objetivo del Proyecto

El propósito de este toolkit es **aprender de forma práctica** cómo estructurar un análisis bursátil profesional en Python.  
A través de ejemplos guiados, se busca entender paso a paso cómo:

- Descargar datos históricos de activos financieros.  
- Limpiar y preparar series de precios.  
- Calcular métricas como retornos, volatilidad o tendencias.  
- Visualizar resultados de forma clara e intuitiva.  
- Aplicar simulaciones como el método de **Monte Carlo**.  

Todo ello utilizando **buenas prácticas de programación**, separación del código en módulos y un entorno reproducible basado en Jupyter Notebooks.

---

## 🧠 Descripción General

El **Toolkit de Análisis Bursátil** está formado por dos componentes principales:

1. **Librería Python (`toolkit_bursatil/`)**  
   Contiene las clases y funciones principales que encapsulan la lógica del análisis bursátil.  
   Ejemplos: manejo de series de precios, conexión con Yahoo Finance o simulaciones estadísticas.

2. **Notebooks (`notebooks/`)**  
   Espacios de trabajo interactivos donde se prueban y documentan los resultados paso a paso, ideal para fines educativos o de investigación.

---

## 🚀 Características

- **Exploración de datos**: notebooks interactivos para descubrir patrones y relaciones.  
- **Series temporales**: análisis estadístico y visualización de precios históricos.  
- **Descarga automática de datos**: conexión directa con Yahoo Finance mediante `yfinance`.  
- **Módulos reutilizables**: clases y funciones listas para importar desde otros proyectos.  
- **Simulaciones**: herramientas para evaluar el comportamiento futuro de activos mediante métodos probabilísticos.  
- **Visualizaciones pedagógicas**: gráficos claros con **Matplotlib** y **Plotly**.  
- **Diseño estructurado**: el código está organizado para que sea fácil de leer, entender y ampliar.

---

## 📁 Estructura del Proyecto

```
01-toolkit-analisis-bursatil/
├── notebooks/
│   ├── 01_exploracion_datos.ipynb      # Introducción y análisis exploratorio
│   └── 02_pruebas_price_series.ipynb   # Ejemplos de análisis de series de precios
├── src/
│   └── toolkit_bursatil/               # Código fuente de la librería
├── requirements.txt                    # Lista de dependencias
├── .gitignore                          # Archivos ignorados por Git
└── README.md                           # Este archivo
```

---

## 🛠️ Instalación

### Requisitos Previos

- **Python 3.8 o superior**  
- **pip** (gestor de paquetes de Python)  
- **Jupyter Notebook** o **JupyterLab**

### Pasos de Instalación

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/diegomuGit/01-toolkit-analisis-bursatil.git
   cd 01-toolkit-analisis-bursatil
   ```

2. **Crear y activar un entorno virtual** (recomendado)

   ```bash
   python -m venv venv

   # En Windows
   venv\Scripts\activate

   # En macOS/Linux
   source venv/bin/activate
   ```

3. **Instalar las dependencias**

   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Uso del Proyecto

### Ejecutar los Notebooks

1. Inicia Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Abre los archivos en la carpeta `notebooks/`:
   - `01_exploracion_datos.ipynb`: análisis inicial y exploración.  
   - `02_pruebas_price_series.ipynb`: análisis de precios y métricas estadísticas.

### Usar el Toolkit desde tu Código

```python
from toolkit_bursatil.data.provider_yahoo import YahooSerie

# Descargar una serie de precios
serie = YahooSerie(ticker="AAPL", start_date="2023-01-01", end_date="2024-12-31")
price_series = serie.get_serie_precios()

# Mostrar información y gráficos
price_series.info()
price_series.plots_report()
```

---

## 📦 Dependencias Principales

| Librería | Uso principal |
|-----------|----------------|
| **pandas** | Manipulación y análisis de datos |
| **numpy** | Cálculo numérico |
| **yfinance** | Descarga de datos financieros |
| **matplotlib** | Visualización tradicional |
| **plotly** | Gráficos interactivos |
| **scikit-learn** | Modelado y análisis estadístico |
| **statsmodels** | Análisis de series temporales |
| **jupyter** | Entorno de trabajo interactivo |

Consulta `requirements.txt` para la lista completa.

---

## 📊 Ejemplos Rápidos

### Obtener Datos de Acciones

```python
import yfinance as yf
import pandas as pd

data = yf.download("MSFT", start="2023-01-01", end="2024-12-31")
print(data.head())
```

### Análisis Básico de Precios

```python
returns = data['Close'].pct_change().dropna()
print("Retorno medio:", returns.mean())
print("Volatilidad:", returns.std())
```

### Visualización de la Serie

```python
import matplotlib.pyplot as plt

data['Close'].plot(title="Precio de Cierre - MSFT")
plt.show()
```

---

## 🤝 Contribuciones

Este proyecto está abierto a mejoras, tanto en código como en documentación.  
Si quieres contribuir:

1. Haz un *fork* del repositorio.  
2. Crea una rama para tus cambios (`git checkout -b feature/NuevaFeature`).  
3. Realiza tus modificaciones y haz *commit*.  
4. Envía un *pull request*.

---

## 📝 Licencia

Este proyecto se distribuye bajo licencia **MIT**.  
Consulta el archivo `LICENSE` para más información.

---

## 👤 Autor

**Diego Muñoz**  
🔗 [GitHub: @diegomuGit](https://github.com/diegomuGit)

---

## 🙏 Agradecimientos

- A **Yahoo Finance**, por ofrecer datos financieros gratuitos y accesibles.  
- A la **comunidad de Python**, por crear herramientas abiertas para el análisis de datos.  
- A todos los estudiantes y colaboradores que impulsan el aprendizaje abierto.

---

⭐ **Si este proyecto te resulta útil o educativo, considera dejarle una estrella en GitHub.**

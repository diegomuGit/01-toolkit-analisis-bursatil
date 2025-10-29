import numpy as np
import pandas as pd
import pytest

from toolkit_bursatil.core.montecarlo import MonteCarloSimulacion


class DummyAsset:
    def __init__(self, mean: float, std: float, name: str = "Dummy"):
        self.mean = mean
        self.std = std
        self.name = name


@pytest.fixture
def simulacion():
    asset = DummyAsset(mean=0.001, std=0.01)
    return MonteCarloSimulacion(objeto=asset, n_sim=10, horizonte=5, valor_inicial=100, seed=42)


def test_ejecutar_generates_dataframe_with_expected_shape(simulacion):
    resultados = simulacion.ejecutar()

    assert isinstance(resultados, pd.DataFrame)
    assert resultados.shape == (simulacion.horizonte, simulacion.n_sim)

    np.random.seed(simulacion.seed)
    retornos = np.random.normal(simulacion.objeto.mean, simulacion.objeto.std, (simulacion.horizonte, simulacion.n_sim))
    esperados = simulacion.valor_inicial * np.exp(pd.DataFrame(retornos).cumsum())

    pd.testing.assert_frame_equal(resultados, esperados)


def test_ejecutar_raises_error_when_objeto_missing_attributes():
    class SinAtributos:
        pass

    simulacion = MonteCarloSimulacion(objeto=SinAtributos())

    with pytest.raises(ValueError):
        simulacion.ejecutar()


def test_resumen_outputs_statistics(simulacion, capsys):
    simulacion.ejecutar()

    simulacion.resumen()
    captured = capsys.readouterr().out

    assert "Resultados Simulación Monte Carlo" in captured
    assert "Media final" in captured
    assert "P5" in captured
    assert "P95" in captured


def test_graficar_calls_show(simulacion, monkeypatch):
    simulacion.ejecutar()

    called = False

    def fake_show():
        nonlocal called
        called = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

    simulacion.graficar(n=3)

    assert called

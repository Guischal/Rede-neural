"""Testes da rede neural de regressao implementada com NumPy."""

import unittest
import numpy as np

from src.rede_neural import RedeNeural
from main import metricas_regressao


class TestRedeNeural(unittest.TestCase):
    def test_gradientes_por_diferencas_finitas(self):
        gerador = np.random.default_rng(7)
        X = gerador.normal(size=(12, 3))
        y = gerador.normal(size=(12, 1))
        passo = 1e-6
        for ativacao in ("relu", "sigmoid"):
            with self.subTest(ativacao=ativacao):
                rede = RedeNeural(semente=9, ativacao_oculta=ativacao)
                rede.forward(X)
                gradientes = rede.backpropagation(X, y)
                for nome, analitico in zip(("W1", "b1", "W2", "b2"), gradientes):
                    parametro = getattr(rede, nome)
                    numerico = np.zeros_like(parametro)
                    for indice in np.ndindex(parametro.shape):
                        original = parametro[indice]
                        parametro[indice] = original + passo
                        mais = rede.calcular_perda(y, rede.forward(X))
                        parametro[indice] = original - passo
                        menos = rede.calcular_perda(y, rede.forward(X))
                        parametro[indice] = original
                        numerico[indice] = (mais - menos) / (2 * passo)
                    np.testing.assert_allclose(analitico, numerico, atol=1e-7, rtol=1e-4)

    def test_treinamento_reproduzivel_e_perda_decrescente(self):
        X = np.random.default_rng(3).normal(size=(100, 3))
        y = (X[:, :1] * 0.7 - X[:, 1:2] * 0.2)
        for ativacao in ("relu", "sigmoid"):
            a, b = RedeNeural(42, ativacao), RedeNeural(42, ativacao)
            a.treinar(X, y, epocas=300, taxa_aprendizado=0.01)
            b.treinar(X, y, epocas=300, taxa_aprendizado=0.01)
            np.testing.assert_array_equal(a.historico_perda, b.historico_perda)
            self.assertLess(a.historico_perda[-1], a.historico_perda[0])
            self.assertTrue(np.isfinite(a.prever(X)).all())

    def test_metricas_regressao(self):
        metricas = metricas_regressao([100, 200, 300], [110, 190, 300])
        self.assertAlmostEqual(metricas["mae"], 20 / 3)
        self.assertGreater(metricas["rmse"], 0)
        self.assertLessEqual(metricas["r2"], 1)


if __name__ == "__main__":
    unittest.main()

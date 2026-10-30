"""Rede neural 3 -> 4 -> 1 para regressao, implementada manualmente com NumPy."""

import numpy as np


class RedeNeural:
    def __init__(self, semente=42, ativacao_oculta="relu"):
        if ativacao_oculta not in ("relu", "sigmoid"):
            raise ValueError("A ativacao oculta deve ser 'relu' ou 'sigmoid'.")
        self.ativacao_oculta = ativacao_oculta
        gerador = np.random.default_rng(semente)
        self.W1 = gerador.normal(0, np.sqrt(2 / 3), (3, 4))
        self.b1 = np.zeros((1, 4))
        self.W2 = gerador.normal(0, np.sqrt(1 / 4), (4, 1))
        self.b2 = np.zeros((1, 1))
        self.historico_perda = []

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def derivada_relu(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivada_sigmoid(ativacao):
        return ativacao * (1 - ativacao)

    def forward(self, X):
        """Retorna uma saida linear: a nota padronizada estimada."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = (self.relu(self.z1) if self.ativacao_oculta == "relu"
                   else self.sigmoid(self.z1))
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.z2  # camada de saida linear para regressao
        return self.a2

    @staticmethod
    def calcular_perda(y, previsoes):
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        previsoes = np.asarray(previsoes, dtype=float).reshape(-1, 1)
        return float(np.mean((previsoes - y) ** 2))

    def backpropagation(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        # Derivada do MSE em relacao a saida linear.
        dz2 = 2 * (self.a2 - y) / len(X)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        derivada = (self.derivada_relu(self.z1) if self.ativacao_oculta == "relu"
                    else self.derivada_sigmoid(self.a1))
        dz1 = (dz2 @ self.W2.T) * derivada
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def atualizar_pesos(self, gradientes, taxa_aprendizado):
        dW1, db1, dW2, db2 = gradientes
        self.W1 -= taxa_aprendizado * dW1
        self.b1 -= taxa_aprendizado * db1
        self.W2 -= taxa_aprendizado * dW2
        self.b2 -= taxa_aprendizado * db2

    def treinar(self, X, y, epocas=2000, taxa_aprendizado=0.01):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != 3 or len(X) == 0 or len(X) != len(y):
            raise ValueError("X deve ter formato (n, 3) e y deve conter n alvos.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Entradas e alvos devem conter somente valores finitos.")
        if epocas < 1 or not np.isfinite(taxa_aprendizado) or taxa_aprendizado <= 0:
            raise ValueError("Epocas e taxa de aprendizado devem ser positivas.")
        self.historico_perda = []
        for _ in range(epocas):
            previsoes = self.forward(X)
            self.historico_perda.append(self.calcular_perda(y, previsoes))
            gradientes = self.backpropagation(X, y)
            self.atualizar_pesos(gradientes, taxa_aprendizado)
        return self.historico_perda

    def prever(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
            raise ValueError("Informe tres notas numericas e validas.")
        return self.forward(X)

"""Carrega o modelo salvo e estima a nota de Matematica de novos usuarios."""

from pathlib import Path

import numpy as np

from src.rede_neural import RedeNeural

CAMINHO_MODELO = Path(__file__).resolve().parents[1] / "models" / "rede_neural.npz"


def carregar_modelo(caminho=CAMINHO_MODELO):
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(
            "Modelo treinado nao encontrado. Execute primeiro: python main.py --treinar"
        )
    with np.load(caminho, allow_pickle=False) as dados:
        ativacao = str(dados["ativacao"].item())
        rede = RedeNeural(ativacao_oculta=ativacao)
        for nome in ("W1", "b1", "W2", "b2"):
            setattr(rede, nome, dados[nome].copy())
        artefato = {
            "media_x": dados["media_x"].copy(),
            "desvio_x": dados["desvio_x"].copy(),
            "media_y": float(dados["media_y"].item()),
            "desvio_y": float(dados["desvio_y"].item()),
            "mae": float(dados["mae"].item()),
            "rmse": float(dados["rmse"].item()),
            "r2": float(dados["r2"].item()),
            "mediana_y": float(dados["mediana_y"].item()),
            "min_y": float(dados["min_y"].item()),
            "max_y": float(dados["max_y"].item()),
        }
    return rede, artefato


def prever_nota(nota_cn, nota_ch, nota_lc, rede=None, artefato=None):
    if rede is None or artefato is None:
        rede, artefato = carregar_modelo()
    entradas = np.array([[nota_cn, nota_ch, nota_lc]], dtype=float)
    if not np.isfinite(entradas).all() or (entradas < 0).any():
        raise ValueError("As tres notas devem ser numeros validos e nao negativos.")
    entradas_padronizadas = (entradas - artefato["media_x"]) / artefato["desvio_x"]
    previsao_padronizada = float(rede.prever(entradas_padronizadas)[0, 0])
    nota_estimada = previsao_padronizada * artefato["desvio_y"] + artefato["media_y"]
    return max(0.0, nota_estimada)

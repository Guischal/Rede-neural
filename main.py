"""Prepara dados, treina, compara e salva a rede neural de regressao do ENEM."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.preparar_dados import COLUNAS, SAIDA, preparar_dados
from src.rede_neural import RedeNeural

RAIZ = Path(__file__).resolve().parent
CAMINHO_MODELO = RAIZ / "models" / "rede_neural.npz"


def metricas_regressao(y_real, y_previsto):
    y_real = np.asarray(y_real, dtype=float).ravel()
    y_previsto = np.asarray(y_previsto, dtype=float).ravel()
    erros = y_real - y_previsto
    mae = float(np.mean(np.abs(erros)))
    rmse = float(np.sqrt(np.mean(erros ** 2)))
    soma_total = float(np.sum((y_real - np.mean(y_real)) ** 2))
    r2 = float(1 - np.sum(erros ** 2) / soma_total) if soma_total > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def salvar_modelo(rede, media_x, desvio_x, media_y, desvio_y, metricas, y_treino):
    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CAMINHO_MODELO,
        W1=rede.W1, b1=rede.b1, W2=rede.W2, b2=rede.b2,
        ativacao=np.array(rede.ativacao_oculta),
        media_x=media_x, desvio_x=desvio_x,
        media_y=np.array(media_y), desvio_y=np.array(desvio_y),
        mae=np.array(metricas["mae"]), rmse=np.array(metricas["rmse"]),
        r2=np.array(metricas["r2"]), mediana_y=np.array(np.median(y_treino)),
        min_y=np.array(np.min(y_treino)), max_y=np.array(np.max(y_treino)),
    )
    print(f"Modelo salvo em: {CAMINHO_MODELO}")


def treinar_rede(epocas=2000, taxa_aprendizado=0.01, semente=42, comparar=False):
    if not SAIDA.is_file():
        raise FileNotFoundError("A amostra nao existe. Prepare os dados primeiro com --csv.")
    dados = pd.read_csv(SAIDA, usecols=COLUNAS)[COLUNAS]
    valores = dados.to_numpy(dtype=float)
    if len(dados) < 10 or not np.isfinite(valores).all() or (valores < 0).any():
        raise ValueError("A amostra precisa ter ao menos 10 registros com notas validas.")

    # Divide antes de calcular estatisticas: o conjunto de teste nao vaza para o treino.
    indices = np.random.default_rng(semente).permutation(len(dados))
    corte = int(0.8 * len(dados))
    treino, teste = valores[indices[:corte]], valores[indices[corte:]]
    X_treino_bruto, X_teste_bruto = treino[:, :3], teste[:, :3]
    y_treino_bruto, y_teste_bruto = treino[:, 3:4], teste[:, 3:4]

    media_x = X_treino_bruto.mean(axis=0)
    desvio_x = X_treino_bruto.std(axis=0)
    desvio_x[desvio_x == 0] = 1.0
    media_y = float(y_treino_bruto.mean())
    desvio_y = float(y_treino_bruto.std())
    if desvio_y == 0:
        desvio_y = 1.0

    X_treino = (X_treino_bruto - media_x) / desvio_x
    X_teste = (X_teste_bruto - media_x) / desvio_x
    y_treino = (y_treino_bruto - media_y) / desvio_y
    y_teste = (y_teste_bruto - media_y) / desvio_y

    print(f"Treino: {len(treino)}; teste: {len(teste)}")
    print(f"Media de Matematica no treino: {media_y:.2f}")
    ativacoes = ("relu", "sigmoid") if comparar else ("relu",)
    redes = {}
    resultados = {}

    for ativacao in ativacoes:
        rede = RedeNeural(semente=semente, ativacao_oculta=ativacao)
        rede.treinar(X_treino, y_treino, epocas, taxa_aprendizado)
        previsao_treino_pad = rede.prever(X_treino)
        previsao_teste_pad = rede.prever(X_teste)
        previsao_teste = previsao_teste_pad * desvio_y + media_y
        metricas = metricas_regressao(y_teste_bruto, previsao_teste)
        resultados[ativacao] = metricas
        redes[ativacao] = rede
        perda_final = rede.calcular_perda(y_treino, previsao_treino_pad)
        print(f"\nAtivacao oculta: {ativacao}")
        print(f"MSE final de treino (escala padronizada): {perda_final:.5f}")
        print(f"MAE no teste: {metricas['mae']:.2f} pontos")
        print(f"RMSE no teste: {metricas['rmse']:.2f} pontos")
        print(f"R2 no teste: {metricas['r2']:.4f}")

    baseline = np.full_like(y_teste_bruto, media_y)
    metricas_baseline = metricas_regressao(y_teste_bruto, baseline)
    print("\nBaseline (prever sempre a media de Matematica do treino):")
    print(f"MAE: {metricas_baseline['mae']:.2f} pontos")
    print(f"RMSE: {metricas_baseline['rmse']:.2f} pontos")
    print(f"R2: {metricas_baseline['r2']:.4f}")

    # ReLU permanece como modelo padrao, preservando o comportamento original.
    rede_escolhida = redes["relu"]
    salvar_modelo(rede_escolhida, media_x, desvio_x, media_y, desvio_y,
                  resultados["relu"], y_treino_bruto)

    if comparar:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figura, eixo = plt.subplots()
        for ativacao, rede in redes.items():
            eixo.plot(np.arange(1, epocas + 1), rede.historico_perda, label=ativacao)
        eixo.set(xlabel="Epoca", ylabel="MSE de treino (alvo padronizado)",
                 title="Comparacao das ativacoes ocultas")
        eixo.legend()
        eixo.grid(alpha=0.3)
        figura.tight_layout()
        caminho = RAIZ / "results" / "comparacao_ativacoes.png"
        caminho.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(caminho, dpi=150)
        plt.close(figura)
        print(f"Grafico salvo em: {caminho}")
    return rede_escolhida


def main():
    parser = argparse.ArgumentParser(description="Rede neural de regressao para estimar nota de Matematica do ENEM 2024.")
    modos = parser.add_mutually_exclusive_group()
    modos.add_argument("--csv", default=os.getenv("ENEM_CSV"), help="Caminho de RESULTADOS_2024.csv.")
    modos.add_argument("--treinar", action="store_true", help="Treina e salva o modelo de regressao.")
    modos.add_argument("--comparar", action="store_true", help="Compara ReLU e Sigmoid e salva o modelo ReLU.")
    parser.add_argument("--codificacao", help="Opcional: força a codificacao, ex.: latin-1.")
    parser.add_argument("--separador", help="Opcional: força o separador, ex.: ';'.")
    parser.add_argument("--epocas", type=int, default=2000)
    parser.add_argument("--taxa-aprendizado", type=float, default=0.01)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()
    if not args.treinar and not args.comparar and not args.csv:
        parser.error("Informe --csv, --treinar ou --comparar.")
    try:
        if args.treinar or args.comparar:
            treinar_rede(args.epocas, args.taxa_aprendizado, args.semente, args.comparar)
        else:
            preparar_dados(args.csv, codificacao=args.codificacao, separador=args.separador)
    except (OSError, ValueError, LookupError, csv.Error) as erro:
        parser.exit(1, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()

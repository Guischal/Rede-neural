"""Prepara uma amostra dos microdados sem alterar o arquivo original."""

import codecs
import csv
from pathlib import Path

import numpy as np
import pandas as pd

COLUNAS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT"]
SAIDA = Path(__file__).resolve().parents[1] / "data" / "enem_amostra.csv"


def detectar_formato(caminho, codificacao=None, separador=None):
    with caminho.open("rb") as arquivo:
        trecho = arquivo.read(1024 * 1024)
    if not trecho:
        raise ValueError("O CSV esta vazio.")
    if codificacao is None:
        try:
            codecs.getincrementaldecoder("utf-8-sig")().decode(trecho, final=False)
            codificacao = "utf-8-sig"
        except UnicodeDecodeError:
            codificacao = "latin-1"
    cabecalho = trecho.decode(codificacao, errors="replace").splitlines()[0]
    if separador is None:
        separador = csv.Sniffer().sniff(cabecalho, delimiters=";,\t|").delimiter
    nomes = next(csv.reader([cabecalho], delimiter=separador))
    faltantes = set(COLUNAS) - set(nomes)
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(sorted(faltantes))}")
    return codificacao, separador


def preparar_dados(caminho_csv, tamanho_amostra=10_000, semente=42,
                   tamanho_bloco=50_000, codificacao=None, separador=None):
    caminho = Path(caminho_csv).expanduser().resolve()
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV nao encontrado: {caminho}")
    if caminho == SAIDA.resolve():
        raise ValueError("A origem deve ser diferente do arquivo de saida.")
    if not 1 <= tamanho_amostra <= 10_000 or tamanho_bloco < 1:
        raise ValueError("A amostra deve ter de 1 a 10.000 linhas; bloco positivo.")
    codificacao, separador = detectar_formato(caminho, codificacao, separador)
    print(f"Leitura: codificacao={codificacao}, separador={separador!r}")
    gerador = np.random.default_rng(semente)
    amostra = pd.DataFrame(columns=COLUNAS, dtype=float)
    prioridades = np.empty(0)
    total_lido = total_valido = 0
    with pd.read_csv(caminho, usecols=COLUNAS, encoding=codificacao,
                     sep=separador, chunksize=tamanho_bloco, dtype=str) as blocos:
        for bloco in blocos:
            total_lido += len(bloco)
            for coluna in COLUNAS:
                bloco[coluna] = pd.to_numeric(
                    bloco[coluna].str.replace(",", ".", regex=False), errors="coerce"
                )
            bloco = bloco[COLUNAS]
            validas = np.isfinite(bloco).all(axis=1) & (bloco >= 0).all(axis=1)
            bloco = bloco.loc[validas]
            total_valido += len(bloco)
            candidatas = pd.concat([amostra, bloco], ignore_index=True)
            prioridades = np.concatenate([prioridades, gerador.random(len(bloco))])
            indices = np.argsort(prioridades, kind="stable")[:tamanho_amostra]
            amostra = candidatas.iloc[indices].reset_index(drop=True)
            prioridades = prioridades[indices]
    if total_valido == 0:
        raise ValueError("Nenhum registro com as quatro notas validas foi encontrado.")
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    amostra.to_csv(SAIDA, index=False, encoding="utf-8")
    print(f"Registros lidos: {total_lido}; validos: {total_valido}; amostra: {len(amostra)}")
    print(f"Amostra salva em: {SAIDA}")
    return amostra

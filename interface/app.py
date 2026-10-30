"""Interface grafica local para estimar a nota de Matematica."""

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.inferencia import carregar_modelo, prever_nota

st.set_page_config(page_title="NeuroENEM | Estimativa", page_icon="🧠", layout="centered")

st.title("🧠 NeuroENEM")
st.subheader("Estimativa de nota em Matemática")
st.write(
    "Informe suas notas de Ciências da Natureza, Ciências Humanas e Linguagens. "
    "A rede neural treinada com os microdados do ENEM estimará sua nota em Matemática."
)

try:
    rede, artefato = carregar_modelo()
except FileNotFoundError as erro:
    st.warning(str(erro))
    st.info(
        "Depois de preparar os microdados, execute `python main.py --treinar` "
        "na pasta principal do projeto. Em seguida, abra novamente esta interface."
    )
    st.stop()



with st.form("formulario_notas"):
    st.markdown("### Suas notas")
    col1, col2 = st.columns(2)
    with col1:
        nota_cn = st.number_input("Ciências da Natureza", min_value=0.0, value=600.0, step=1.0)
        nota_lc = st.number_input("Linguagens e Códigos", min_value=0.0, value=600.0, step=1.0)
    with col2:
        nota_ch = st.number_input("Ciências Humanas", min_value=0.0, value=600.0, step=1.0)
    enviar = st.form_submit_button("Estimar minha nota em Matemática", type="primary", use_container_width=True)

if enviar:
    try:
        estimativa = prever_nota(nota_cn, nota_ch, nota_lc, rede, artefato)
        st.divider()
        st.markdown("### Resultado da estimativa")
        st.metric("Nota estimada em Matemática", f"{estimativa:.1f} pontos")
        st.caption(
            f"No conjunto de teste, o erro absoluto médio (MAE) foi de "
            f"{artefato['mae']:.1f} pontos. Esse valor resume o erro médio do modelo "
            "na avaliação; não é uma garantia nem um intervalo individual de confiança."
        )
        if estimativa < artefato["min_y"] or estimativa > artefato["max_y"]:
            st.info(
                "A estimativa ficou fora da faixa de notas de Matemática observada "
                "no conjunto de treinamento. Interprete esse resultado com cautela."
            )
        st.write(
            "**Importante:** esta é uma estimativa estatística baseada em padrões "
            "aprendidos nos dados do ENEM 2024. Não representa uma nota garantida "
            "nem substitui o resultado oficial do exame."
        )
    except ValueError as erro:
        st.error(str(erro))

with st.expander("Sobre o modelo e sua avaliação"):
    st.write("Arquitetura preservada: 3 entradas → 4 neurônios ocultos → 1 saída linear.")
    st.write("A camada oculta usa ReLU no modelo salvo. O treinamento também permite comparar Sigmoid.")
    st.write(f"MAE no teste: {artefato['mae']:.2f} pontos")
    st.write(f"RMSE no teste: {artefato['rmse']:.2f} pontos")
    st.write(f"R² no teste: {artefato['r2']:.4f}")
    st.write(
        "A nota de Matemática é o alvo do treinamento, não uma entrada fornecida "
        "pelo usuário. A padronização é calculada somente com os dados de treino."
    )

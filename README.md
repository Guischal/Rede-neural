# Rede Neural — Estimativa de nota de Matemática do ENEM

1. PASSO A PASSO
    * INSTALAR DEPENDENCIAS NA PASTA RAIZ {python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt}
    
    * ALIMENTAR A REDE COM MICRODADOS (tópico seguinte) {
        .\.venv\Scripts\python.exe main.py --csv "CAMINHO DO DIRETORIO exemplo: C:\downloads\microdadosenem" }
    * TREINAR A REDE NEURAL {.\.venv\Scripts\python.exe main.py --treinar}
    * INICIAR INTERFACE {.\.venv\Scripts\python.exe -m streamlit run interface/app.py}

2. Obter os microdados do ENEM

    Site: https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem/resultados

    Foi utilizado nesse projeto os microdados referentes ao ano de 2022

3. Funcionamento da rede neural

O treinamento usa 80% dos registros para treino e 20% para teste. A média e o desvio das entradas e do alvo são calculados somente no conjunto de treino. O alvo Matemática também é padronizado durante o treinamento; a previsão é convertida novamente para a escala original de pontos.

O modelo ReLU é salvo em `models/rede_neural.npz`. Esse arquivo inclui os pesos, a ativação escolhida, os parâmetros de padronização e as métricas do conjunto de teste.

Para comparar ReLU e Sigmoid:

```powershell
.\.venv\Scripts\python.exe main.py --comparar
```

A comparação salva `results/comparacao_ativacoes.png` e também salva a rede ReLU para uso na interface. O baseline prevê sempre a média de Matemática observada no conjunto de treino.


4. ESTRUTURA

```text
Rede-Neural/
├── data/
│   └── enem_amostra.csv          # gerado localmente
├── src/
│   ├── preparar_dados.py         # preparação dos microdados
│   ├── rede_neural.py            # rede neural manual 3 → 4 → 1
│   └── inferencia.py             # carrega modelo e estima novas notas
├── interface/
│   └── app.py                    # interface Streamlit
├── models/
│   └── rede_neural.npz           # gerado após o treinamento
├── results/
│   └── comparacao_ativacoes.png  # gerado no modo comparação
├── tests/
│   └── test_rede_neural.py
├── main.py
└── requirements.txt
```
*main.py - executa preparação treinamento e comparação
*src/rede_neural.py - implementação da rede neural
*src/preparar_dados.py - preparamento dos microdados, leitura e seleção dos dados presentes no csv
*src/inferencia.py - carrega o modelo e realiza estimativas
*interface/app.py - interface gráfica com biblioteca streamlit
*tests/test_rede_neural.py - testes automatizados
*models/rede_neural.npz - armazena o modelo treinado
*data/enem_amostra.csv - dados preparados para treinamento








ignore "st.success(f"Modelo carregado. Ativação oculta: {rede.ativacao_oculta.upper()}")"
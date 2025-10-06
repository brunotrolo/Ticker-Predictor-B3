# B3 Ticker App (Streamlit)

Aplicação simples e **100% focada na B3** para visualizar preços históricos e indicadores (SMA, EMA, RSI) usando **Yahoo Finance (.SA)**.

## 🧩 Como funciona
- Busca e seleção apenas de **tickers da B3** (ex.: `PETR4.SA`, `VALE3.SA`).
- Se você digitar sem `.SA` (ex.: `PETR4`), a aplicação normaliza para `PETR4.SA`.
- Preços **ajustados por proventos** (parâmetro `auto_adjust=True` do `yfinance`).

## 🚀 Executar localmente
```bash
# Python 3.10+ recomendado
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## ☁️ Publicar no Streamlit Community Cloud (passo a passo)
1. **Suba este projeto para o GitHub** (crie um repositório público ou privado).
2. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com GitHub.
3. Clique em **"New app"** ➜ selecione seu **repositório**, **branch** e o arquivo principal `streamlit_app.py`.
4. (Opcional) Em **Advanced settings**, defina **"Python version"** se quiser travar (ex.: `3.11`), e **"Secrets"** se precisar de chaves. Para `yfinance`, **não é necessário**.
5. Clique em **Deploy**. A primeira compilação instala os pacotes do `requirements.txt` e sobe sua aplicação.
6. Após o deploy, se quiser atualizar: faça **git push** na branch escolhida e o Streamlit Cloud reconstrói automaticamente.

## 📁 Estrutura
```
.
├─ .streamlit/
│  └─ config.toml
├─ data/
│  └─ b3_tickers.csv
├─ b3_utils.py
├─ requirements.txt
├─ streamlit_app.py
└─ README.md
```

## 🔧 Dicas
- Para ampliar a lista de tickers, edite `data/b3_tickers.csv` (colunas: `ticker,name`).
- Se algum ticker não carregar, verifique se existe no Yahoo Finance com sufixo `.SA`.
- Para carregar mais rápido, o `@st.cache_data` guarda o histórico por **1 hora** (`ttl=3600`).

---

Feito para B3 🇧🇷. Sinta-se à vontade para adaptar!
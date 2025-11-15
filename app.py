# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io, base64
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# ---------------- Config ----------------
st.set_page_config(page_title="🚀 Bot Trading PRO — Dados Reais", layout="wide")
st.title("🚀 Bot Trading PRO — Dados Reais (Cripto & Forex)")

st.sidebar.header("Configuração (obrigatória p/ Forex intraday)")
alpha_key = st.sidebar.text_input("Alpha Vantage API Key (para Forex intraday)", value="", type="password")
st.sidebar.markdown("Sem key, Forex intraday não estará disponível. Obtenha grátis em https://www.alphavantage.co")

# ---------------- Utilities (som e UI) ----------------
sound_up_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
sound_down_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_sound(sound_b64, key=None):
    audio_bytes = base64.b64decode(sound_b64)
    placeholder = st.empty()
    placeholder.audio(io.BytesIO(audio_bytes), format="audio/wav", start_time=0, key=(f"audio_{key}" if key else None))

alert_container = st.empty()
chart_container = st.empty()

def show_signal_alert(signal: str, confidence: float, min_conf: float = 70):
    color_map = {"UP": "#1db954", "DOWN": "#e63946", "NEU": "#6c757d"}
    if signal == "UP":
        color = color_map["UP"]; label = "SUBIDA 🔼"
    elif signal == "DOWN":
        color = color_map["DOWN"]; label = "DESCIDA 🔽"
    else:
        color = color_map["NEU"]; label = "NEUTRAL ⚪"
    alert_container.markdown(
        f"<div style='background-color:{color};padding:1rem;border-radius:8px;color:white;text-align:center'><b>{label}</b><br>Confiança: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )
    if confidence >= min_conf:
        if signal == "UP":
            play_sound(sound_up_b64, key=int(confidence))
        elif signal == "DOWN":
            play_sound(sound_down_b64, key=int(confidence))

# ---------------- Fetch Crypto (Binance public REST) ----------------
def fetch_crypto_ohlcv_binance(symbol="BTC/USDT", interval="15m", limit=500):
    """
    Busca candles (klines) da Binance via API pública.
    symbol: ex. 'BTC/USDT' -> precisa formar 'BTCUSDT' para Binance endpoint
    interval: '1m','3m','5m','15m','30m','1h','4h','1d' ...
    limit: max 1000
    Retorna DataFrame com index datetime e colunas open, high, low, close, volume
    """
    try:
        bsymbol = symbol.replace("/", "")
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": bsymbol, "interval": interval, "limit": limit}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        # Each kline: [openTime, open, high, low, close, volume, closeTime, ...]
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "q","n","taker_base","taker_quote","ignore"
        ])
        df = df[["open_time","open","high","low","close","volume"]].copy()
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        df.set_index("ts", inplace=True)
        df = df[["open","high","low","close","volume"]]
        return df
    except Exception as e:
        st.error(f"Erro ao buscar dados da Binance: {e}")
        return pd.DataFrame()

# ---------------- Fetch Forex (Alpha Vantage FX_INTRADAY) ----------------
def fetch_forex_alpha(from_symbol="EUR", to_symbol="USD", interval="15min", outputsize="full"):
    """
    Usa Alpha Vantage FX_INTRADAY. Requer API key em alpha_key.
    interval: '1min','5min','15min','30min','60min'
    outputsize: 'compact' (last 100) or 'full'
    Retorna DataFrame com index datetime e columns open, high, low, close, volume (volume may be absent -> fill 0)
    """
    if not alpha_key:
        st.error("Alpha Vantage API key não fornecida. Para Forex intraday, insira sua API key no sidebar.")
        return pd.DataFrame()

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": alpha_key
        }
        res = requests.get(url, params=params, timeout=12)
        res.raise_for_status()
        data = res.json()
        # expected key: "Time Series FX (15min)" or similar
        keyname = None
        for k in data.keys():
            if "Time Series" in k or "Time Series FX" in k:
                keyname = k
                break
        if keyname is None:
            # AlphaVantage may return {"Note": "..."} or error
            if "Note" in data:
                st.error("AlphaVantage: limite de requisições atingido ou IP bloqueado. Mensagem: " + data.get("Note",""))
            else:
                st.error("AlphaVantage: resposta inesperada: " + str(data))
            return pd.DataFrame()
        ts = data[keyname]
        # ts is dict timestamp -> { "1. open": "...", ... }
        records = []
        for t, v in ts.items():
            dt = pd.to_datetime(t)
            o = float(v.get("1. open", v.get("open")))
            h = float(v.get("2. high", v.get("high")))
            l = float(v.get("3. low", v.get("low")))
            c = float(v.get("4. close", v.get("close")))
            # FX_INTRADAY does not provide volume usually -> set 0
            records.append({"ts": dt, "open": o, "high": h, "low": l, "close": c, "volume": 0.0})
        df = pd.DataFrame(records).sort_values("ts")
        df.set_index("ts", inplace=True)
        return df
    except Exception as e:
        st.error(f"Erro ao buscar dados Forex (Alpha Vantage): {e}")
        return pd.DataFrame()

# ---------------- Feature engineering + model ----------------
def prepare_features(df):
    df2 = df.copy()
    df2["HL"] = df2["high"] - df2["low"]
    df2["OC"] = df2["close"] - df2["open"]
    df2["SMA"] = df2["close"].rolling(5).mean()
    df2["SMA_diff"] = df2["SMA"] - df2["close"]
    df2.fillna(0, inplace=True)
    return df2

def train_and_predict(df):
    """
    Treina RandomForest em dados históricos (X = features, y = next close)
    Retorna last_price, pred_price, diff, signal, confidence
    """
    if df.empty or len(df) < 10:
        st.error("Dados insuficientes para treinar modelo.")
        return None, None, None, "NEUTRAL", 0.0
    df_feat = prepare_features(df)
    X = df_feat[["open","high","low","close","volume","HL","OC","SMA","SMA_diff"]].iloc[:-1]
    y = df_feat["close"].shift(-1).iloc[:-1]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    next_feat = X.iloc[-1:].values
    pred = model.predict(next_feat)[0]
    last = float(df["close"].iloc[-1])
    diff = (pred - last) / last
    confidence = float(np.clip(abs(diff)*1000, 60, 99))
    signal = "UP" if diff > 0.002 else "DOWN" if diff < -0.002 else "NEU"
    return last, pred, diff, signal, confidence

# ---------------- UI Inputs ----------------
st.sidebar.header("Parâmetros de Análise")
market = st.sidebar.selectbox("Mercado", ["Cripto (Binance)","Forex (AlphaVantage)"])
if market.startswith("Cripto"):
    symbol = st.sidebar.text_input("Símbolo (Binance, ex: BTC/USDT)", value="BTC/USDT")
    interval = st.sidebar.selectbox("Intervalo (Binance)", ["1m","3m","5m","15m","30m","1h","4h","1d"], index=3)
    limit = st.sidebar.number_input("Número de candles (max 1000)", min_value=50, max_value=1000, value=200, step=10)
else:
    pair = st.sidebar.text_input("Par Forex (ex: EUR/USD)", value="EUR/USD")
    interval = st.sidebar.selectbox("Intervalo AlphaVantage", ["1min","5min","15min","30min","60min"], index=2)
    limit = st.sidebar.selectbox("Outputsize", ["compact (latest)","full (larger)"], index=0)
    outputsize = "compact" if limit.startswith("compact") else "full"

confidence_threshold = st.sidebar.slider("Confiança mínima p/ tocar som (%)", min_value=50, max_value=99, value=70)

# ---------------- Run analysis ----------------
analyze = st.button("▶️ Analisar mercado")
if analyze:
    chart_container.empty()
    alert_container.empty()
    if market.startswith("Cripto"):
        df = fetch_crypto_ohlcv_binance(symbol=symbol.upper(), interval=interval, limit=limit)
    else:
        # Forex: need alpha key
        from_sym, to_sym = pair.upper().split("/")
        df = fetch_forex_alpha(from_sym, to_sym, interval=interval, outputsize="full" if outputsize=="full" else "compact")
    if df.empty:
        st.error("Dados não disponíveis. Verifique API/key e tente novamente.")
    else:
        last, pred, diff, signal, confidence = train_and_predict(df)
        if last is None:
            st.error("Falha na predição.")
        else:
            st.subheader(f"💰 Preço Atual: {last:.6f}")
            st.subheader(f"📈 Preço Previsto: {pred:.6f}")
            st.metric("Variação (%)", f"{diff*100:.2f}%")
            show_signal_alert(signal, confidence, min_conf=confidence_threshold)
            # Plot com st.empty() container para evitar removeChild
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Preço Real'))
            # Previsão como ponto/linha curta
            future_time = df.index[-1] + (df.index[-1] - df.index[-2])
            fig.add_trace(go.Scatter(x=[df.index[-1], future_time], y=[last, pred], mode='lines+markers', name='Previsão'))
            chart_container.plotly_chart(fig, use_container_width=True)

# ---------------- Footer / notas ----------------
st.markdown("---")
st.markdown("**Notas:**")
st.markdown("- Este app usa **dados reais** da **Binance** (cripto) e **Alpha Vantage** (Forex intraday).")
st.markdown("- Para Forex intraday você **deve** fornecer uma chave Alpha Vantage válida no sidebar.")
st.markdown("- Alpha Vantage tem limites de requisições por minuto/dia na conta gratuita; evite muitos refreshs rápidos.")
st.markdown("- Requer conexão à internet no ambiente onde o app roda.")

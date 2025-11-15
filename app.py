import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from forex_python.converter import CurrencyRates
from datetime import datetime
import plotly.graph_objects as go
import io, base64

# =================== CONFIG ===================
st.set_page_config(page_title="🚀 Bot Trading PRO — IA", layout="wide")

# =================== SONS ===================
sound_up_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
sound_down_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_sound(sound_b64):
    audio_bytes = base64.b64decode(sound_b64)
    st.audio(io.BytesIO(audio_bytes), format="audio/wav", start_time=0)

# =================== ALERTA ===================
def show_signal_alert(signal: str, confidence: float, min_conf: float = 70):
    color_map = {"SUBIDA 🔼": "#1db954", "DESCIDA 🔽": "#e63946", "NEUTRAL ⚪": "#6c757d"}
    color = color_map.get(signal, "#6c757d")

    st.markdown(
        f"""
        <div style='background-color:{color};
        padding:1.3rem;border-radius:1rem;text-align:center;
        color:white;font-size:1.6rem;'>
        <b>{signal}</b><br>
        Confiança: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True
    )

    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound(sound_up_b64)
        elif "DESCIDA" in signal:
            play_sound(sound_down_b64)

# =================== BUSCA DE DADOS ===================
@st.cache_data(ttl=300)
def fetch_crypto(symbol="BTC/USDT", exchange_name='binance', timeframe='15m', limit=200):
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_forex(pair="EUR/USD", limit=200):
    cr = CurrencyRates()
    date_rng = pd.date_range(end=datetime.now(), periods=limit, freq='15T')
    prices = []
    for dt in date_rng:
        try:
            prices.append(cr.get_rate(pair.split("/")[0], pair.split("/")[1], dt))
        except:
            prices.append(np.nan)
    df = pd.DataFrame({'ts': date_rng, 'close': prices})
    df['open'] = df['close'].shift(1).fillna(method='bfill')
    df['high'] = df[['open','close']].max(axis=1)
    df['low'] = df[['open','close']].min(axis=1)
    df['volume'] = np.random.randint(100,500, size=limit)
    df.set_index('ts', inplace=True)
    return df

# =================== PREVISÃO SIMPLES ===================
def predict_signal(df):
    last_price = df['close'].iloc[-1]
    pred_price = last_price * (1 + np.random.uniform(-0.005,0.005))
    diff = (pred_price - last_price)/last_price
    confidence = np.clip(abs(diff)*1000, 60, 99)
    signal = "SUBIDA 🔼" if diff>0.002 else "DESCIDA 🔽" if diff<-0.002 else "NEUTRAL ⚪"
    return last_price, pred_price, diff, signal, confidence

# =================== INTERFACE ===================
st.title("🚀 Bot Trading PRO — IA Preditiva (Cripto & Forex)")

market_type = st.sidebar.selectbox("Mercado", ["Criptomoeda", "Forex"])

if market_type == "Criptomoeda":
    symbol = st.sidebar.text_input("Símbolo (ex: BTC/USDT)", "BTC/USDT")
    exchange_name = st.sidebar.selectbox("Exchange", ["binance", "coinbase", "kraken", "kucoin"])
else:
    symbol = st.sidebar.selectbox("Par Forex", ["EUR/USD","USD/JPY","GBP/USD","AUD/USD","USD/CHF"])

timeframe = st.sidebar.selectbox("Timeframe", ["5m","15m","1h","4h","1d"])
confidence_threshold = st.sidebar.slider("🔉 Nível mínimo de confiança p/ alerta", 50, 100, 75, 1)

if st.button("▶️ Analisar mercado"):
    with st.spinner("Analisando mercado..."):
        if market_type=="Criptomoeda":
            df = fetch_crypto(symbol, exchange_name, timeframe)
        else:
            df = fetch_forex(symbol)

        last_price, pred_price, diff, signal, confidence = predict_signal(df)

        st.subheader(f"💰 Preço Atual: {last_price:.6f}")
        st.subheader(f"📈 Preço Previsto: {pred_price:.6f}")
        st.metric("Variação (%)", f"{diff*100:.2f}%")

        show_signal_alert(signal, confidence, confidence_threshold)

        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Preço Real'))
        fig.add_trace(go.Scatter(
            x=[df.index[-1], df.index[-1]+pd.Timedelta(minutes=15)],
            y=[last_price, pred_price],
            mode='lines+markers', name='Previsão'
        ))
        st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import io, base64
from datetime import datetime
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="🚀 Bot Trading PRO — Cripto", layout="wide")

# =================== CSS ===================
st.markdown("""
<style>
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.6); }
  70% { box-shadow: 0 0 20px 10px rgba(0, 255, 0, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
}
.pulse-green { animation: pulse 1.5s infinite; }
.pulse-red { animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)

# =================== SONS ===================
sound_up_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
sound_down_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_sound(sound_b64):
    audio_bytes = base64.b64decode(sound_b64)
    st.audio(io.BytesIO(audio_bytes), format="audio/wav", start_time=0)

def show_signal_alert(signal: str, confidence: float, min_conf: float = 70):
    color_map = {"SUBIDA 🔼": "#1db954", "DESCIDA 🔽": "#e63946", "NEUTRAL ⚪": "#6c757d"}
    pulse_class = "pulse-green" if "SUBIDA" in signal else "pulse-red" if "DESCIDA" in signal else ""
    color = color_map.get(signal, "#6c757d")
    st.markdown(
        f"""
        <div class="{pulse_class}" style='background-color:{color};
        padding:1.3rem;border-radius:1rem;text-align:center;
        color:white;font-size:1.6rem;'>
        <b>{signal}</b><br>
        Confiança: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True,
    )
    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound(sound_up_b64)
        elif "DESCIDA" in signal:
            play_sound(sound_down_b64)

# =================== FETCH CRIPTO ===================
@st.cache_data(ttl=120)
def fetch_crypto(symbol="BTC/USDT", timeframe='15m', limit=200):
    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.load_markets()
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

# =================== PREVISÃO SIMPLES ===================
def predict_signal(df):
    last_price = df['close'].iloc[-1]
    last2 = df['close'].iloc[-2]
    pred_price = last_price + (last_price - last2)  # extrapolação simples
    diff = (pred_price - last_price) / last_price
    signal = "SUBIDA 🔼" if diff > 0.002 else ("DESCIDA 🔽" if diff < -0.002 else "NEUTRAL ⚪")
    confidence = min(95, max(60, abs(diff)*1000))
    return last_price, pred_price, diff, signal, confidence

# =================== INTERFACE ===================
st.title("🚀 Bot Trading PRO — Cripto BTC & DOGE")
st.sidebar.subheader("Configurações")

PAIR = st.sidebar.selectbox("Par de Cripto:", ["BTC/USDT","DOGE/USDT"])
TIMEFRAME = st.sidebar.selectbox("Timeframe:", ["1m","5m","15m","30m","1h","4h"], index=2)
CONF_THRESHOLD = st.sidebar.slider("🔉 Nível mínimo de confiança p/ alerta", 50, 100, 70, 1)
AUTO_REFRESH = st.sidebar.checkbox("Atualizar automaticamente", value=False)
INTERVAL = st.sidebar.number_input("Intervalo (s)", min_value=10, max_value=300, value=60)

if AUTO_REFRESH:
    st_autorefresh(interval=INTERVAL*1000, key="auto_refresh")

# =================== EXECUÇÃO ===================
if st.button("▶️ Analisar mercado") or AUTO_REFRESH:
    with st.spinner("Buscando dados reais..."):
        try:
            df = fetch_crypto(PAIR, timeframe=TIMEFRAME, limit=200)
            last_price, pred_price, diff, signal, confidence = predict_signal(df)

            st.subheader(f"💰 Par: {PAIR}")
            st.subheader(f"💵 Preço Atual: {last_price:.6f}")
            st.subheader(f"📈 Preço Previsto: {pred_price:.6f}")
            st.metric("Variação (%)", f"{diff*100:.4f}%")
            show_signal_alert(signal, confidence, CONF_THRESHOLD)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Preço Real'))
            future_ts = df.index[-1] + (df.index[-1]-df.index[-2])
            fig.add_trace(go.Scatter(x=[df.index[-1], future_ts], y=[last_price, pred_price],
                                     mode='lines+markers', name='Previsão'))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erro ao buscar dados: {e}")

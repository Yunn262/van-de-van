import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import io, base64
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="🚀 Bot Trading Pro — IA Preditiva", layout="wide")

# =================== CSS ===================
st.markdown("""
<style>
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(0,255,0,0.6); }
  70% { box-shadow: 0 0 20px 10px rgba(0,255,0,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,255,0,0); }
}
.pulse-green { animation: pulse 1.5s infinite; }
.pulse-red { animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)

# =================== FUNÇÕES ===================
@st.cache_data(ttl=180)
def fetch_crypto(symbol, exchange_name='binance', timeframe='15m', limit=200):
    try:
        exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        exchange.load_markets()
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        return df
    except Exception:
        return None

@st.cache_data(ttl=180)
def fetch_forex(symbol='EUR/USD', timeframe='15m', limit=200):
    try:
        exchange = ccxt.oanda()
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        return df
    except Exception:
        return None

# Sons
sound_up_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
sound_down_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_sound(sound_b64):
    audio_bytes = base64.b64decode(sound_b64)
    st.audio(io.BytesIO(audio_bytes), format="audio/wav", start_time=0)

# =================== ANALISADOR ===================
def analyze_market(df):
    last = df['close'].iloc[-1]
    last2 = df['close'].iloc[-2]

    # Tendência recente
    trend = ((last - last2) / last2) * 100

    # Previsão super simples (candle momentum)
    pred = last + (last - last2)

    # Diferença percentual
    diff = ((pred - last) / last) * 100

    # Sensibilidade aprimorada
    if diff > 0.05:
        signal = "SUBIDA 🔼"
    elif diff < -0.05:
        signal = "DESCIDA 🔽"
    else:
        if trend > 0:
            signal = "SUBIDA 🔼"
        elif trend < 0:
            signal = "DESCIDA 🔽"
        else:
            signal = "NEUTRAL ⚪"

    # Confiança dinâmica
    confidence = min(99, max(50, abs(diff) * 8 + 60))

    return last, pred, diff, signal, confidence

# =================== ALERTA ===================
def show_signal_alert(signal, confidence, min_conf):
    color_map = {"SUBIDA 🔼": "#1db954", "DESCIDA 🔽": "#e63946", "NEUTRAL ⚪": "#6c757d"}
    pulse_class = "pulse-green" if "SUBIDA" in signal else "pulse-red" if "DESCIDA" in signal else ""

    st.markdown(
        f"""
        <div class="{pulse_class}" style="
            background-color:{color_map[signal]};
            padding:1.3rem;border-radius:1rem;
            text-align:center;color:white;
            font-size:1.6rem;">
            <b>{signal}</b><br>
            Confiança: {confidence:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Som
    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound(sound_up_b64)
        elif "DESCIDA" in signal:
            play_sound(sound_down_b64)

# =================== INTERFACE ===================
st.title("🚀 Bot Trading Pro — IA Preditiva (Cripto & Forex)")

market_type = st.sidebar.selectbox("Mercado", ["Criptomoeda", "Forex"])

if market_type == "Criptomoeda":
    symbol = st.sidebar.text_input("Símbolo (ex: BTC/USDT)", "BTC/USDT")
    exchange_name = st.sidebar.selectbox("Exchange", ["binance", "kucoin", "kraken"])
else:
    symbol = st.sidebar.selectbox("Par Forex", ["EUR/USD","GBP/USD","USD/JPY","AUD/USD"])

timeframe = st.sidebar.selectbox("Timeframe", ["5m","15m","1h","4h","1d"])
confidence_threshold = st.sidebar.slider("🔉 Nível mínimo de confiança p/ alerta", 50, 100, 70)
auto_refresh = st.sidebar.checkbox("Atualização automática")
interval = st.sidebar.number_input("Intervalo (segundos)", min_value=10, max_value=300, value=60)

if auto_refresh:
    st_autorefresh(interval=interval*1000, key="refresh_key")

# =================== EXECUÇÃO ===================
if st.button("Analisar Mercado") or auto_refresh:
    with st.spinner("🔍 Buscando dados reais..."):
        if market_type == "Criptomoeda":
            df = fetch_crypto(symbol, exchange_name, timeframe)
        else:
            df = fetch_forex(symbol, timeframe)

        if df is None:
            st.error("❌ Erro ao buscar dados reais. O par pode não existir na API selecionada.")
        else:
            last, pred, diff, signal, confidence = analyze_market(df)

            st.subheader(f"💰 Preço Atual: {last:.5f}")
            st.subheader(f"📈 Preço Previsto: {pred:.5f}")
            st.metric("Variação (%)", f"{diff:.3f}%")

            show_signal_alert(signal, confidence, confidence_threshold)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import io, base64, math
from datetime import datetime
import plotly.graph_objects as go

# TTS
from gtts import gTTS

# ----------------- Config -----------------
st.set_page_config(page_title="🚀 Bot PRO — Dark + Painel Avançado", layout="wide")
# Dark theme CSS (modo escuro total, neon accents)
st.markdown("""
<style>
html, body, .stApp, .main, .block-container {
  background: #0b0f14;
  color: #e6eef6;
}
h1, h2, h3, h4, h5 { color: #ffffff; }
.stButton>button { background: linear-gradient(90deg,#0b8cff,#6b4bff); color: white; }
div[role="radiogroup"] label, .stSelectbox, .stTextInput, .stNumberInput {
  color: #e6eef6;
}
.card {
  background:#071025; border-radius:10px; padding:12px; box-shadow: 0 4px 12px rgba(0,0,0,0.6);
}
.small-muted { color: #9fb7d8; font-size:12px; }
.metric-container { background: #071025; padding:10px; border-radius:8px; }
.legend-up { color: #1bd77a; font-weight:700; }
.legend-down { color: #ff6b6b; font-weight:700; }
.heatcell { padding:6px; border-radius:6px; color:white; text-align:center; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ----------------- Helpers: sounds / TTS -----------------
# tiny base64 beeps as fallback
SOUND_UP = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
SOUND_DOWN = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_beep(b64):
    try:
        audio_bytes = base64.b64decode(b64)
        st.audio(io.BytesIO(audio_bytes), format="audio/wav")
    except Exception:
        pass

def speak_ptbr(text: str):
    """TTS with gTTS — wrapped to avoid breaking if fails."""
    try:
        tts = gTTS(text=text, lang='pt-br')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        st.audio(fp, format="audio/mp3")
    except Exception as e:
        # show a small warning but do not crash
        st.experimental_set_query_params()  # noop to avoid linter error
        st.warning("TTS indisponível no ambiente (ignorado).")

# ----------------- CCXT helpers -----------------
def build_exchange(exch_name: str):
    # sanitize names for ccxt
    mapping = {"coinbase": "coinbasepro", "kucoin": "kucoin", "kraken": "kraken", "binance":"binance"}
    key = mapping.get(exch_name.lower(), exch_name.lower())
    try:
        exc_cls = getattr(ccxt, key)
        return exc_cls({'enableRateLimit': True})
    except Exception:
        return None

@st.cache_data(ttl=90)
def fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str='15m', limit: int=200):
    exc = build_exchange(exchange_name)
    if exc is None:
        return None, f"Exchange {exchange_name} não encontrada."
    try:
        exc.load_markets()
        data = exc.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df, None
    except Exception as e:
        return None, str(e)

# ----------------- Indicators -----------------
def sma(series, n):
    return series.rolling(n).mean()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def volatility(series, window=14):
    # historical volatility: std of log returns annualized (approx)
    logret = np.log(series / series.shift(1)).dropna()
    vol = logret.rolling(window).std() * np.sqrt(365 * 24 * 60 / (15))  # approximate if 15m timeframe
    return vol

def trend_strength(series, short=5, long=20):
    s_short = sma(series, short)
    s_long = sma(series, long)
    strength = ((s_short - s_long) / s_long) * 100
    return strength

# ----------------- UI Inputs -----------------
st.sidebar.header("Configurações")
exchange = st.sidebar.selectbox("Exchange", ["binance", "coinbase", "kraken", "kucoin"])
symbol = st.sidebar.text_input("Par (ex: BTC/USDT)", value="BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m","3m","5m","15m","30m","1h","4h"], index=3)
limit = st.sidebar.slider("Quantidade de candles", min_value=50, max_value=1000, value=300, step=10)
autorefresh = st.sidebar.checkbox("Atualizar automaticamente", value=False)
interval = st.sidebar.number_input("Intervalo (s) - auto refresh", min_value=10, max_value=600, value=60)
confidence_threshold = st.sidebar.slider("Nível mínimo de confiança p/ voz/alerta (%)", 50, 99, 70)

# Favorites heatmap input (comma-separated)
fav_input = st.sidebar.text_area("Pares favoritos (virgula separados) — p/ heatmap", value="BTC/USDT,ETH/USDT", height=60)
favorites = [s.strip().upper() for s in fav_input.split(",") if s.strip()][:10]

# auto-refresh key
refresh_key = f"rf_{exchange}_{symbol.replace('/','_')}"
if autorefresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=interval * 1000, key=refresh_key)

# main layout
left, right = st.columns([2,1])

with left:
    st.markdown("## 📊 Painel Avançado")
    # analyze button
    analyze = st.button("▶️ Analisar mercado")
    if analyze or autorefresh:
        # fetch data
        df, err = fetch_ohlcv(exchange, symbol, timeframe=timeframe, limit=int(limit))
        if err or df is None or df.empty:
            st.error(f"Erro ao buscar dados: {err or 'Dados vazios / par inválido.'}")
        else:
            # compute indicators
            df['rsi'] = rsi(df['close'], period=14)
            macd_line, macd_signal, macd_hist = macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            vol = volatility(df['close'], window=14)
            df['volatility'] = vol
            df['trend_strength'] = trend_strength(df['close'], short=5, long=20)

            # analysis numbers
            last = float(df['close'].iloc[-1])
            prev = float(df['close'].iloc[-2])
            last_rsi = float(df['rsi'].iloc[-1])
            macd_h = float(df['macd_hist'].iloc[-1])
            vol_now = float(df['volatility'].iloc[-1]) if not math.isnan(df['volatility'].iloc[-1]) else 0.0
            trend_now = float(df['trend_strength'].iloc[-1])

            # compute diff% from prev candle
            diff = ((last - prev) / prev) * 100.0
            # confidence heuristic: combine abs(diff), RSI distance from 50, MACD hist magnitude
            conf = min(99.0, max(50.0, abs(diff)*6 + abs(last_rsi-50)*0.6 + abs(macd_h)*8))
            # decide signal
            if diff > 0:
                signal = "SUBIDA 🔼"
            elif diff < 0:
                signal = "DESCIDA 🔽"
            else:
                signal = "NEUTRAL ⚪"

            # top metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Preço Atual", f"{last:.8f}", delta=f"{diff:.4f}%")
            col2.metric("RSI (14)", f"{last_rsi:.2f}", delta=f"{(last_rsi-50):+.2f}")
            col3.metric("MACD Hist", f"{macd_h:.6f}")
            col4.metric("Volatilidade (inst.)", f"{vol_now:.6f}")

            # trend strength and textual
            st.markdown(f"**Força da Tendência:** {trend_now:.3f}%  •  **Confiança:** {conf:.1f}%")
            # show alert, sound and TTS if above threshold
            show_box = st.empty()
            show_box.markdown("")  # placeholder
            # visual alert
            color_map = {"SUBIDA 🔼":"#1bd77a","DESCIDA 🔽":"#ff6b6b","NEUTRAL ⚪":"#9fb7d8"}
            st.markdown(
                f"""<div style='background:{color_map[signal]};padding:10px;border-radius:8px;color:#061219;text-align:center;font-weight:700'>
                {signal} — Confiança: {conf:.1f}%</div>""",
                unsafe_allow_html=True
            )
            # play beep and speak
            if conf >= confidence_threshold:
                if "SUBIDA" in signal:
                    play_beep(SOUND_UP)
                    speak_ptbr("Alerta de subida! Alta provável.")
                elif "DESCIDA" in signal:
                    play_beep(SOUND_DOWN)
                    speak_ptbr("Alerta de queda! Baixa provável.")
                else:
                    speak_ptbr("Mercado neutro no momento.")

            # plot price + MACD + RSI subplots
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='#00c0ff')))
            # add predicted short projection (momentum)
            future_ts = df.index[-1] + (df.index[-1] - df.index[-2])
            pred = last + (last - prev)
            fig.add_trace(go.Scatter(x=[df.index[-1], future_ts], y=[last, pred],
                                     mode='lines+markers', name='Projeção', line=dict(color='#ffd24d')))
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor='#071025', plot_bgcolor='#071025')
            st.plotly_chart(fig, use_container_width=True)

            # MACD & RSI small charts
            colm1, colm2 = st.columns(2)
            with colm1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df.index, y=df['macd_hist'], name='MACD Hist', line=dict(color='#ff7b7b')))
                fig2.update_layout(height=220, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor='#071025', plot_bgcolor='#071025')
                st.plotly_chart(fig2, use_container_width=True)
            with colm2:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='#9ad4ff')))
                fig3.update_layout(height=220, margin=dict(l=10,r=10,t=20,b=10), yaxis=dict(range=[0,100]), paper_bgcolor='#071025', plot_bgcolor='#071025')
                st.plotly_chart(fig3, use_container_width=True)

with right:
    st.markdown("## 🔎 Heatmap de Favoritos")
    heatcells = []
    if not favorites:
        st.info("Adicione pares favoritos na sidebar para ver o heatmap.")
    else:
        # compute strength metric for each favorite
        grid = []
        for s in favorites:
            try:
                df_s, err = fetch_ohlcv(exchange, s, timeframe=timeframe, limit=200)
                if df_s is None or err:
                    grid.append((s, None))
                    continue
                last = float(df_s['close'].iloc[-1])
                prev = float(df_s['close'].iloc[-6]) if len(df_s) > 6 else float(df_s['close'].iloc[-2])
                pct = ((last - prev) / prev) * 100
                r = float(rsi(df_s['close']).iloc[-1]) if 'close' in df_s else 50.0
                score = pct * 2 + (r - 50) * 0.3  # heuristic
                grid.append((s, score))
            except Exception:
                grid.append((s, None))
        # show sorted
        grid_sorted = sorted(grid, key=lambda x: (x[1] is None, -(x[1] or 0)))
        for s, score in grid_sorted:
            if score is None:
                st.markdown(f"<div class='card small-muted'>{s}: <span style='color:#f5c97b'>indisponível</span></div>", unsafe_allow_html=True)
            else:
                # color scale neon: green -> red
                val = np.clip(score, -10, 10)
                # map to color
                if val >= 3:
                    color = "#12c56e"
                elif val >= 0:
                    color = "#a6ff4d"
                elif val >= -3:
                    color = "#ffae42"
                else:
                    color = "#ff5c5c"
                st.markdown(f"<div class='card heatcell' style='background:{color}'>{s}<br>{score:.3f}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔧 Opções rápidas")
    st.markdown("- Atualize o símbolo na sidebar para qualquer par suportado pela exchange.")
    st.markdown("- Heatmap avalia até 10 pares favoritos (separados por vírgula).")

# footer
st.markdown("<hr style='border-color:#123'>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Observações: TTS pode falhar em alguns ambientes; ajustes manuais de símbolo podem ser necessários por exchange.</div>", unsafe_allow_html=True)

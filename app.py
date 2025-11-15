# app.py
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import io, base64
from datetime import datetime
import plotly.graph_objects as go

# TTS
from gtts import gTTS

# =================== Config página ===================
st.set_page_config(page_title="🚀 Bot Trading PRO — Cripto Multi-Exchange", layout="wide")
st.title("🚀 Bot Trading PRO — Cripto (Multi-Exchange)")

# =================== CSS simples ===================
st.markdown("""
<style>
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.6); }
  70% { box-shadow: 0 0 20px 10px rgba(0, 255, 0, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
}
.pulse-green { animation: pulse 1.2s infinite; }
.pulse-red { animation: pulse 1.2s infinite; }
</style>
""", unsafe_allow_html=True)

# =================== Containers (evitar removeChild) ===================
chart_container = st.empty()
alert_container = st.empty()
status_container = st.empty()

# =================== SONS (curtos em base64) ===================
sound_up_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
sound_down_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_sound_b64(sound_b64):
    audio_bytes = base64.b64decode(sound_b64)
    st.audio(io.BytesIO(audio_bytes), format="audio/wav", start_time=0)

def speak_ptbr(text: str):
    """
    Gera TTS em pt-BR com gTTS e toca no Streamlit.
    Requer gTTS no requirements.txt.
    """
    try:
        tts = gTTS(text=text, lang='pt-br')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        st.audio(mp3_fp, format="audio/mp3")
    except Exception as e:
        # se TTS falhar, apenas ignora (mas mostra aviso no status)
        status_container.warning(f"TTS falhou: {str(e)}")

# =================== Fetch dados (ccxt) ===================
def build_exchange(exchange_id: str):
    """
    Retorna instância ccxt do exchange_id.
    Faz alguns mapeamentos: coinbase -> coinbasepro (quando necessário).
    """
    try:
        # Map common naming issues
        if exchange_id.lower() == "coinbase":
            # coinbase pro class name in ccxt is coinbasepro
            return ccxt.coinbasepro({'enableRateLimit': True})
        # normal case (binance, kraken, kucoin)
        exchange_cls = getattr(ccxt, exchange_id)
        return exchange_cls({'enableRateLimit': True})
    except AttributeError:
        # última tentativa: lowercase attribute
        try:
            exchange_cls = getattr(ccxt, exchange_id.lower())
            return exchange_cls({'enableRateLimit': True})
        except Exception:
            return None
    except Exception:
        return None

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str='15m', limit: int=300):
    """
    Busca OHLCV via ccxt. Retorna DataFrame ou None em caso de erro.
    """
    exc = build_exchange(exchange_id)
    if exc is None:
        return None, f"Exchange '{exchange_id}' não suportada."
    try:
        # Carregar markets para validar simbolo
        exc.load_markets()
        data = exc.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        # garantir floats
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df, None
    except ccxt.ExchangeNotAvailable as e:
        return None, f"Exchange indisponível: {e}"
    except ccxt.NetworkError as e:
        return None, f"Erro de rede: {e}"
    except ccxt.BadSymbol as e:
        return None, f"Par inválido nesta exchange: {symbol}"
    except Exception as e:
        return None, f"Erro ao buscar dados: {e}"

# =================== Previsão / regra simples ===================
def analyze_df(df: pd.DataFrame):
    """
    Simples análise: extrapola último movimento, calcula diff% e sinal com sensibilidade.
    Retorna: last, pred, diff_pct, signal, confidence
    """
    if df is None or df.empty or len(df) < 3:
        return None, None, None, "NEUTRAL ⚪", 0.0
    last = float(df['close'].iloc[-1])
    last2 = float(df['close'].iloc[-2])
    # extrapolação (momentum curto)
    pred = last + (last - last2)
    diff_pct = ((pred - last) / last) * 100
    # sensibilidade
    if diff_pct > 0.05:
        signal = "SUBIDA 🔼"
    elif diff_pct < -0.05:
        signal = "DESCIDA 🔽"
    else:
        # fallback para tendência
        trend = ((last - last2) / last2) * 100
        signal = "SUBIDA 🔼" if trend > 0 else ("DESCIDA 🔽" if trend < 0 else "NEUTRAL ⚪")
    confidence = float(min(99, max(50, abs(diff_pct) * 8 + 60)))
    return last, pred, diff_pct, signal, confidence

def show_signal(signal: str, confidence: float, min_conf: float):
    alert_container.empty()
    color_map = {"SUBIDA 🔼": "#1db954", "DESCIDA 🔽": "#e63946", "NEUTRAL ⚪": "#6c757d"}
    color = color_map.get(signal, "#6c757d")
    pulse_class = "pulse-green" if "SUBIDA" in signal else "pulse-red" if "DESCIDA" in signal else ""
    alert_container.markdown(
        f"""
        <div class="{pulse_class}" style='background-color:{color};
        padding:1.2rem;border-radius:10px;text-align:center;color:white;font-size:1.4rem;'>
        <b>{signal}</b><br>
        Confiança: {confidence:.1f}%
        </div>
        """, unsafe_allow_html=True)
    # sons + voz
    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound_b64(sound_up_b64)
            speak_ptbr("Sinal de subida detectado. Alta provável.")
        elif "DESCIDA" in signal:
            play_sound_b64(sound_down_b64)
            speak_ptbr("Sinal de queda detectado. Baixa provável.")
        else:
            # se quiser falar neutro com confiança alta
            speak_ptbr("Mercado neutro no momento.")

# =================== UI Inputs ===================
st.sidebar.header("Configurações")
exchange_choice = st.sidebar.selectbox("Exchange:", ["binance", "coinbase", "kraken", "kucoin"])
symbol = st.sidebar.text_input("Par (ex: BTC/USDT)", value="BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe:", ["1m","3m","5m","15m","30m","1h","4h","1d"], index=3)
limit = st.sidebar.number_input("Quantidade de candles (max 1000)", min_value=50, max_value=1000, value=300, step=10)
confidence_threshold = st.sidebar.slider("Confiança mínima p/ tocar som (%)", 50, 99, 70)
auto_refresh = st.sidebar.checkbox("Atualizar automaticamente", value=False)
interval = st.sidebar.number_input("Intervalo (s) - auto refresh", min_value=10, max_value=600, value=60)

# unique refresh key per symbol/exchange to reduce DOM conflict
refresh_key = f"refresh_{exchange_choice}_{symbol.replace('/', '_')}"

if auto_refresh:
    # import aqui para evitar não usado se não for selecionado
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=interval * 1000, key=refresh_key)

# botão de análise
if st.button("▶️ Analisar mercado") or auto_refresh:
    status_container.info("Buscando dados reais — aguarde...")
    df, err = fetch_ohlcv(exchange_choice, symbol, timeframe=timeframe, limit=int(limit))
    if err:
        status_container.error(err)
        st.stop()
    if df is None or df.empty:
        status_container.error("Dados vazios ou par não encontrado.")
        st.stop()

    # análise
    last, pred, diff_pct, signal, confidence = analyze_df(df)

    # mostrar resultados
    st.subheader(f"Par: {symbol}  •  Exchange: {exchange_choice}")
    st.write(f"Último candle: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
    st.metric("Preço Atual", f"{last:.8f}")
    st.metric("Variação Prevista (%)", f"{diff_pct:.6f}%")
    show_signal(signal, confidence, confidence_threshold)

    # gráfico (usa container)
    chart_container.empty()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Preço Real'))
    future_ts = df.index[-1] + (df.index[-1] - df.index[-2])
    fig.add_trace(go.Scatter(x=[df.index[-1], future_ts], y=[last, pred], mode='lines+markers', name='Previsão'))
    fig.update_layout(title=f"{symbol} — {exchange_choice}", margin=dict(l=10,r=10,t=30,b=10))
    chart_container.plotly_chart(fig, use_container_width=True)

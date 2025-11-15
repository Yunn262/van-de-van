# Bot Trading PRO — Cripto Multi-Exchange (Versão Aprimorada)

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import io, base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# TTS
from gtts import gTTS

# =================== Config página ===================
st.set_page_config(page_title="🚀 Bot Trading PRO — Cripto Multi-Exchange", layout="wide")
st.title("🚀 Bot Trading PRO — Cripto (Multi-Exchange)")

# =================== CSS melhorado ===================
st.markdown("""
<style>
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.6); }
  70% { box-shadow: 0 0 20px 10px rgba(0, 255, 0, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
}
.pulse-green { animation: pulse 1.2s infinite; }
.pulse-red { animation: pulse 1.2s infinite; }
.metric-container {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.signal-container {
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    text-align: center;
    color: white;
    font-weight: bold;
}
.analysis-container {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# =================== Containers ===================
chart_container = st.empty()
alert_container = st.empty()
status_container = st.empty()
metrics_container = st.empty()
indicators_container = st.empty()

# =================== SONS (base64) ===================
# Sons simplificados para garantir funcionamento
sound_up_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
sound_down_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_sound_b64(sound_b64):
    try:
        audio_bytes = base64.b64decode(sound_b64)
        st.audio(io.BytesIO(audio_bytes), format="audio/wav", start_time=0, autoplay=True)
    except Exception as e:
        status_container.warning(f"Erro ao reproduzir som: {str(e)}")

def speak_ptbr(text: str):
    """
    Gera TTS em pt-BR com gTTS e toca no Streamlit.
    """
    try:
        tts = gTTS(text=text, lang='pt-br')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        st.audio(mp3_fp, format="audio/mp3", autoplay=True)
    except Exception as e:
        status_container.warning(f"TTS falhou: {str(e)}")

# =================== Fetch dados (ccxt) ===================
def build_exchange(exchange_id: str):
    """
    Retorna instância ccxt do exchange_id.
    """
    try:
        # Mapeamento para exchanges específicas
        exchange_map = {
            "coinbase": "coinbasepro",
            "kucoin": "kucoin",
            "binance": "binance",
            "kraken": "kraken",
            "bybit": "bybit",
            "huobi": "huobi"
        }
        
        normalized_id = exchange_id.lower()
        if normalized_id in exchange_map:
            exchange_id = exchange_map[normalized_id]
            
        exchange_cls = getattr(ccxt, exchange_id)
        return exchange_cls({'enableRateLimit': True})
    except AttributeError:
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

# =================== Indicadores Técnicos ===================
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(df, period=20, std_dev=2):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_ema(df, period=20):
    return df['close'].ewm(span=period).mean()

def calculate_sma(df, period=20):
    return df['close'].rolling(window=period).mean()

# =================== Análise Avançada ===================
def analyze_df(df: pd.DataFrame, use_indicators=True):
    """
    Análise avançada com indicadores técnicos.
    Retorna: last, pred, diff_pct, signal, confidence, indicators
    """
    if df is None or df.empty or len(df) < 3:
        return None, None, None, "NEUTRO ⚪", 0.0, {}
    
    # Preços básicos
    last = float(df['close'].iloc[-1])
    last2 = float(df['close'].iloc[-2])
    last3 = float(df['close'].iloc[-3])
    
    # Previsão simples baseada em momentum
    pred = last + (last - last2) * 0.8 + (last2 - last3) * 0.2
    diff_pct = ((pred - last) / last) * 100
    
    # Calcular indicadores se solicitado
    indicators = {}
    if use_indicators:
        # RSI
        indicators['rsi'] = calculate_rsi(df).iloc[-1]
        
        # MACD
        macd, signal_line, histogram = calculate_macd(df)
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_histogram'] = histogram.iloc[-1]
        
        # Bandas de Bollinger
        upper_band, sma, lower_band = calculate_bollinger_bands(df)
        indicators['bb_upper'] = upper_band.iloc[-1]
        indicators['bb_middle'] = sma.iloc[-1]
        indicators['bb_lower'] = lower_band.iloc[-1]
        
        # EMA e SMA
        indicators['ema_20'] = calculate_ema(df, 20).iloc[-1]
        indicators['sma_50'] = calculate_sma(df, 50).iloc[-1]
        
        # Volume médio
        indicators['volume_avg'] = df['volume'].rolling(window=14).mean().iloc[-1]
        indicators['volume_current'] = df['volume'].iloc[-1]
    
    # Análise combinada para sinal
    signal_strength = 0
    
    # Análise de preço
    if diff_pct > 0.1:
        signal_strength += 2
    elif diff_pct > 0.05:
        signal_strength += 1
    elif diff_pct < -0.1:
        signal_strength -= 2
    elif diff_pct < -0.05:
        signal_strength -= 1
    
    # Análise de RSI se disponível
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        if rsi < 30:  # Sobrevendido
            signal_strength += 2
        elif rsi > 70:  # Sobrecomprado
            signal_strength -= 2
    
    # Análise de MACD se disponível
    if 'macd_histogram' in indicators:
        macd_hist = indicators['macd_histogram']
        if macd_hist > 0:
            signal_strength += 1
        else:
            signal_strength -= 1
    
    # Análise de Bandas de Bollinger se disponível
    if 'bb_lower' in indicators and 'bb_upper' in indicators:
        if last < indicators['bb_lower']:
            signal_strength += 1
        elif last > indicators['bb_upper']:
            signal_strength -= 1
    
    # Análise de Volume se disponível
    if 'volume_current' in indicators and 'volume_avg' in indicators:
        if indicators['volume_current'] > indicators['volume_avg'] * 1.5:
            signal_strength += 1
    
    # Determinar sinal baseado na força combinada
    if signal_strength >= 3:
        signal = "FORTE SUBIDA 🚀"
    elif signal_strength >= 1:
        signal = "SUBIDA 🔼"
    elif signal_strength <= -3:
        signal = "FORTE DESCIDA 📉"
    elif signal_strength <= -1:
        signal = "DESCIDA 🔽"
    else:
        signal = "NEUTRO ⚪"
    
    # Calcular confiança baseada na força do sinal e na volatilidade
    confidence = float(min(99, max(50, abs(signal_strength) * 10 + 50)))
    
    return last, pred, diff_pct, signal, confidence, indicators

def show_signal(signal: str, confidence: float, min_conf: float):
    alert_container.empty()
    
    # Mapeamento de cores e classes CSS
    color_map = {
        "FORTE SUBIDA 🚀": "#00d084", 
        "SUBIDA 🔼": "#1db954", 
        "FORTE DESCIDA 📉": "#d63031", 
        "DESCIDA 🔽": "#e63946", 
        "NEUTRO ⚪": "#6c757d"
    }
    color = color_map.get(signal, "#6c757d")
    
    # Determinar classe de pulsação
    if "SUBIDA" in signal:
        pulse_class = "pulse-green"
    elif "DESCIDA" in signal:
        pulse_class = "pulse-red"
    else:
        pulse_class = ""
    
    # Exibir sinal
    alert_container.markdown(
        f"""
        <div class="{pulse_class} signal-container" style='background-color:{color};'>
        <h2>{signal}</h2>
        <p>Confiança: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sons + voz se confiança for alta
    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound_b64(sound_up_b64)
            speak_ptbr(f"Sinal de {signal.lower()} detectado com alta confiança. Compra recomendada.")
        elif "DESCIDA" in signal:
            play_sound_b64(sound_down_b64)
            speak_ptbr(f"Sinal de {signal.lower()} detectado com alta confiança. Venda recomendada.")
        else:
            speak_ptbr("Mercado neutro no momento. Aguarde uma oportunidade melhor.")

def show_indicators(indicators):
    if not indicators:
        return
        
    indicators_container.empty()
    with indicators_container.container():
        st.subheader("Indicadores Técnicos")
        
        # Criar colunas para os indicadores
        col1, col2, col3 = st.columns(3)
        
        # RSI
        if 'rsi' in indicators:
            rsi_value = indicators['rsi']
            rsi_color = "green" if rsi_value < 30 else ("red" if rsi_value > 70 else "orange")
            col1.metric("RSI (14)", f"{rsi_value:.2f}", None)
            col1.progress(min(100, max(0, rsi_value)))
        
        # MACD
        if 'macd_histogram' in indicators:
            macd_hist = indicators['macd_histogram']
            macd_signal = "Cruzamento de Compra" if macd_hist > 0 else "Cruzamento de Venda"
            col2.metric("MACD", f"{macd_hist:.6f}", macd_signal)
        
        # Bandas de Bollinger
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            bb_position = (indicators['bb_upper'] - indicators['bb_lower']) / 2
            col3.metric("Posição nas Bandas", f"{bb_position:.4f}", None)
        
        # EMA/SMA
        if 'ema_20' in indicators and 'sma_50' in indicators:
            ema_sma_diff = indicators['ema_20'] - indicators['sma_50']
            trend_signal = "Tendência de Alta" if ema_sma_diff > 0 else "Tendência de Baixa"
            col1.metric("EMA 20 vs SMA 50", f"{ema_sma_diff:.4f}", trend_signal)
        
        # Volume
        if 'volume_current' in indicators and 'volume_avg' in indicators:
            vol_ratio = indicators['volume_current'] / indicators['volume_avg']
            vol_signal = "Alto Volume" if vol_ratio > 1.5 else ("Baixo Volume" if vol_ratio < 0.5 else "Volume Normal")
            col2.metric("Volume Atual vs Média", f"{vol_ratio:.2f}x", vol_signal)
        
        # Preço vs Bandas de Bollinger
        if 'bb_upper' in indicators and 'bb_lower' in indicators and 'bb_middle' in indicators:
            bb_width = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] * 100
            col3.metric("Largura das Bandas", f"{bb_width:.2f}%", None)

def show_metrics(last, pred, diff_pct, confidence, df):
    metrics_container.empty()
    with metrics_container.container():
        # Criar métricas em colunas
        col1, col2, col3, col4 = st.columns(4)
        
        # Preço atual
        col1.metric("Preço Atual", f"{last:.8f}")
        
        # Previsão
        col2.metric("Previsão", f"{pred:.8f}", f"{diff_pct:.4f}%")
        
        # Variação 24h (simulada)
        if len(df) >= 24:
            day_change = ((last - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100
            col3.metric("Variação 24h", f"{day_change:.4f}%")
        else:
            col3.metric("Variação Total", f"{((last - df['close'].iloc[0]) / df['close'].iloc[0]) * 100:.4f}%")
        
        # Confiança
        col4.metric("Confiança", f"{confidence:.1f}%")
        
        # Informações adicionais
        st.write(f"Último candle: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Período analisado: {df.index[0].strftime('%Y-%m-%d %H:%M:%S')} a {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")

# =================== UI Inputs ===================
st.sidebar.header("Configurações")
exchange_choice = st.sidebar.selectbox("Exchange:", ["binance", "coinbase", "kraken", "kucoin", "bybit", "huobi"])
symbol = st.sidebar.text_input("Par (ex: BTC/USDT)", value="BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe:", ["1m","3m","5m","15m","30m","1h","4h","1d"], index=3)
limit = st.sidebar.number_input("Quantidade de candles (max 1000)", min_value=50, max_value=1000, value=300, step=10)
confidence_threshold = st.sidebar.slider("Confiança mínima p/ tocar som (%)", 50, 99, 70)
use_indicators = st.sidebar.checkbox("Usar indicadores técnicos", value=True)
show_volume = st.sidebar.checkbox("Mostrar volume no gráfico", value=True)
auto_refresh = st.sidebar.checkbox("Atualizar automaticamente", value=False)
interval = st.sidebar.number_input("Intervalo (s) - auto refresh", min_value=10, max_value=600, value=60)

# Botão de análise
analyze_button = st.sidebar.button("▶️ Analisar mercado")

# unique refresh key per symbol/exchange
refresh_key = f"refresh_{exchange_choice}_{symbol.replace('/', '_')}"

if auto_refresh:
    # Import aqui para evitar não usado se não for selecionado
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval * 1000, key=refresh_key)
        # Se auto_refresh estiver ativo, analisar automaticamente
        analyze_button = True
    except ImportError:
        st.sidebar.warning("streamlit-autorefresh não instalado. Auto refresh desativado.")

# Lógica principal
if analyze_button:
    status_container.info("Buscando dados reais — aguarde...")
    df, err = fetch_ohlcv(exchange_choice, symbol, timeframe=timeframe, limit=int(limit))
    if err:
        status_container.error(err)
        st.stop()
    if df is None or df.empty:
        status_container.error("Dados vazios ou par não encontrado.")
        st.stop()

    # Análise
    last, pred, diff_pct, signal, confidence, indicators = analyze_df(df, use_indicators)

    # Mostrar métricas
    show_metrics(last, pred, diff_pct, confidence, df)
    
    # Mostrar indicadores técnicos se habilitado
    if use_indicators and indicators:
        show_indicators(indicators)

    # Mostrar sinal
    show_signal(signal, confidence, confidence_threshold)

    # Gráfico melhorado
    chart_container.empty()
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{symbol} — {exchange_choice}", "Volume"),
        row_width=[0.2, 0.7]
    )
    
    # Adicionar candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Preço"
    ), row=1, col=1)
    
    # Adicionar médias móveis se indicadores habilitados
    if use_indicators and indicators:
        if 'ema_20' in indicators:
            ema_20 = calculate_ema(df, 20)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ema_20,
                mode='lines',
                name='EMA 20',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        
        if 'sma_50' in indicators:
            sma_50 = calculate_sma(df, 50)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=sma_50,
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=1)
            ), row=1, col=1)
        
        if 'bb_upper' in indicators and 'bb_lower' in indicators and 'bb_middle' in indicators:
            upper_band, sma, lower_band = calculate_bollinger_bands(df)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=upper_band,
                mode='lines',
                name='Banda Superior',
                line=dict(color='gray', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=lower_band,
                mode='lines',
                name='Banda Inferior',
                line=dict(color='gray', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.2)'
            ), row=1, col=1)
    
    # Adicionar previsão
    future_ts = df.index[-1] + (df.index[-1] - df.index[-2])
    fig.add_trace(go.Scatter(
        x=[df.index[-1], future_ts],
        y=[last, pred],
        mode='lines+markers',
        name='Previsão',
        line=dict(color='red', width=2, dash='dash')
    ), row=1, col=1)
    
    # Adicionar volume se habilitado
    if show_volume:
        colors = ['green' if row['open'] - row['close'] <= 0 else 'red' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ), row=2, col=1)
    
    # Atualizar layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=600
    )
    
    chart_container.plotly_chart(fig, use_container_width=True)
    
    # Opção de exportar dados
    if st.button("📥 Exportar dados CSV"):
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{symbol}_{exchange_choice}_{timeframe}.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# Rodapé com informações
st.sidebar.markdown("---")
st.sidebar.info("Este bot de trading é para fins educacionais. Não se constitui em recomendação de investimento.")

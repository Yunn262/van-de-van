"""
Bot integration Streamlit - "EMA+RSI AUTO 2M" adapted
Support:
 - Binance (via ccxt) - FULL implementation
 - MetaTrader5 (MT5) - example (requires MetaTrader5 terminal + python package)
 - Deriv & IQ Option - placeholders (unofficial APIs; user must provide library/credentials)

Features:
 - Live data from exchange (Binance implemented)
 - Indicator calculation (EMA9/21/50, RSI14, volume)
 - Signal logic (same as you approved)
 - 60s delayed order entry (non-blocking)
 - Colored arrows on Plotly chart and metric cards
 - Safety: basic order sizing, stop loss / take profit params

INSTRUCTIONS:
 - Install required libs: pip install streamlit plotly pandas ccxt ta-meta mt5-connector (or MetaTrader5)
 - Fill API keys in Streamlit sidebar for the exchange you want to use
 - To enable MT5 you must run this script on the same machine with MetaTrader 5 terminal and the Python package installed
 - IQ Option and Deriv require unofficial connectors; this file includes placeholders where you can add your connector logic

USAGE: streamlit run bot_integration_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import threading
import time
import ccxt

# Optional: MT5 import guarded
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

# ----------------- Helper functions -----------------

def fetch_binance_ohlcv(symbol: str, timeframe: str = '2m', limit: int = 200, api_keys=None):
    """Fetch klines using ccxt Binance. If api_keys provided uses private client (for futures/orders)."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    if api_keys:
        exchange.apiKey = api_keys.get('apiKey')
        exchange.secret = api_keys.get('secret')
    # ccxt uses timeframe like '2m'
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# MT5 fetch (example)
def fetch_mt5_ohlcv(symbol: str, timeframe_minutes: int = 2, bars: int = 200):
    if not MT5_AVAILABLE:
        raise RuntimeError('MetaTrader5 package not available')
    # timeframe mapping example
    tf_map = {1: mt5.TIMEFRAME_M1, 2: mt5.TIMEFRAME_M2, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15}
    tf = tf_map.get(timeframe_minutes, mt5.TIMEFRAME_M2)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'tick_volume': 'volume'})
    return df[['time','open','high','low','close','volume']]

# Placeholder for Deriv / IQ Option data fetch - users should replace with their own connector

def fetch_deriv_ticks(symbol: str, limit: int = 200):
    raise NotImplementedError('Deriv fetch not implemented in this template. Plug your own connector.')


def fetch_iqoption_ohlcv(symbol: str, timeframe: str = '2m', limit: int = 200):
    raise NotImplementedError('IQ Option fetch not implemented in this template. Plug your own connector.')

# Indicator calculations

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    # RSI 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    df['ema_distance'] = (df['EMA9'] - df['EMA21']).abs() / df['close']
    # volatility helper
    df['volatility'] = (df['high'] - df['low']) / df['open']
    return df

# Signal logic (returns tuple(signal_text, color, numeric_signal))
def generate_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    signal = 'NONE'
    color = 'gray'
    num = 0
    # CALL
    if (
        last.EMA9 > last.EMA21 and
        last.EMA21 > last.EMA50 and
        last.close > last.EMA9 and
        50 <= last.RSI <= 65 and
        last.vol_up and
        last.ema_distance > 0.0005
    ):
        signal = 'CALL'
        color = 'green'
        num = 1
    # PUT
    elif (
        last.EMA9 < last.EMA21 and
        last.EMA21 < last.EMA50 and
        last.close < last.EMA9 and
        35 <= last.RSI <= 50 and
        last.vol_up and
        last.ema_distance > 0.0005
    ):
        signal = 'PUT'
        color = 'red'
        num = -1
    # RSI safety
    if last.RSI > 70 or last.RSI < 30:
        signal = 'NONE'
        color = 'gray'
        num = 0
    return signal, color, num

# ---------- Order placement functions (examples) ----------

def place_binance_order(symbol: str, side: str, amount: float, api_keys: dict, test=True):
    """Place a market order on Binance using ccxt. For production set test=False and configure properly.
    Note: for futures you need to set "defaultType" : "future" and symbol margin details.
    """
    exchange = ccxt.binance({
        'apiKey': api_keys.get('apiKey'),
        'secret': api_keys.get('secret'),
        'enableRateLimit': True,
    })
    try:
        if test:
            print('TEST ORDER', symbol, side, amount)
            return {'status': 'test', 'symbol': symbol, 'side': side, 'amount': amount}
        order = exchange.create_market_buy_order(symbol, amount) if side == 'buy' else exchange.create_market_sell_order(symbol, amount)
        return order
    except Exception as e:
        st.error(f"Erro ao enviar ordem Binance: {e}")
        return None


def place_mt5_order(symbol: str, order_type: str, volume: float, price=None, sl=None, tp=None):
    if not MT5_AVAILABLE:
        raise RuntimeError('MT5 not available')
    # connect if needed
    if not mt5.initialize():
        raise RuntimeError('MT5 initialize failed')
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError('Symbol not found on MT5')
    # prepare request
    point = symbol_info.point
    deviation = 20
    if order_type.lower() == 'buy':
        price = mt5.symbol_info_tick(symbol).ask
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': mt5.ORDER_TYPE_BUY,
            'price': price,
            'deviation': deviation,
            'magic': 234000,
            'comment': 'EMA_RSI_bot',
        }
    else:
        price = mt5.symbol_info_tick(symbol).bid
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': mt5.ORDER_TYPE_SELL,
            'price': price,
            'deviation': deviation,
            'magic': 234000,
            'comment': 'EMA_RSI_bot',
        }
    result = mt5.order_send(request)
    return result

# Placeholder order functions for IQ Option / Deriv

def place_iqoption_order(*args, **kwargs):
    raise NotImplementedError('IQ Option order placement not implemented in this template.')


def place_deriv_order(*args, **kwargs):
    raise NotImplementedError('Deriv order placement not implemented in this template.')

# ---------- Delayed executor (non-blocking) ----------

def delayed_order_executor(exchange_name: str, order_func, delay_seconds: int, *args, **kwargs):
    """Schedules order_func to run after delay_seconds in background thread."""
    def worker():
        time.sleep(delay_seconds)
        try:
            res = order_func(*args, **kwargs)
            st.session_state['last_order_result'] = res
        except Exception as e:
            st.session_state['last_order_result'] = {'error': str(e)}

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return True

# ----------------- Streamlit App -----------------

st.set_page_config(page_title='EMA+RSI AUTO 2M - Bot Integration', layout='wide')
st.title('EMA+RSI AUTO 2M — Bot Integration')

col1, col2 = st.columns([1, 2])
with col1:
    st.sidebar.header('Exchange / Keys')
    exchange_choice = st.sidebar.selectbox('Exchange', ['Binance', 'MT5', 'Deriv', 'IQ Option'])
    symbol = st.sidebar.text_input('Symbol (Binance format e.g. BTC/USDT or MT5 e.g. EURUSD)', 'BTC/USDT')
    timeframe = st.sidebar.selectbox('Timeframe', ['2m'], index=0)
    api_key = st.sidebar.text_input('API Key')
    api_secret = st.sidebar.text_input('API Secret')
    test_mode = st.sidebar.checkbox('Test mode (no real orders)', value=True)
    risk_percent = st.sidebar.slider('Risk % per trade', min_value=0.1, max_value=5.0, value=1.0)
    amount_fixed = st.sidebar.number_input('Order amount (if you want fixed lot/quantity, 0 = use risk%)', value=0.0, min_value=0.0)
    start_live = st.sidebar.button('Start Live')
    stop_live = st.sidebar.button('Stop Live')

with col2:
    st.write('Live data & chart')
    chart_area = st.empty()
    info_area = st.empty()

if 'live' not in st.session_state:
    st.session_state['live'] = False
if 'last_signal' not in st.session_state:
    st.session_state['last_signal'] = ('NONE', 'gray')
if 'last_order_result' not in st.session_state:
    st.session_state['last_order_result'] = None

if start_live:
    st.session_state['live'] = True
if stop_live:
    st.session_state['live'] = False

# single update function

def update_and_render():
    try:
        if exchange_choice == 'Binance':
            keys = {'apiKey': api_key, 'secret': api_secret} if api_key and api_secret else None
            df = fetch_binance_ohlcv(symbol.replace('/', '/'), timeframe, limit=300, api_keys=keys)
        elif exchange_choice == 'MT5':
            df = fetch_mt5_ohlcv(symbol, timeframe_minutes=2, bars=300)
        elif exchange_choice == 'Deriv':
            st.warning('Deriv connector not implemented. Add your own fetch function.')
            return
        elif exchange_choice == 'IQ Option':
            st.warning('IQ Option connector not implemented. Add your own fetch function.')
            return
        else:
            st.error('Exchange not supported')
            return

        df = add_indicators(df)
        signal, color, num = generate_signal(df)
        st.session_state['last_signal'] = (signal, color)

        # Show metrics + colored arrow
        arrow_html = ''
        if signal == 'CALL':
            arrow_html = "<div style='font-size:28px;color:limegreen'>⬆️</div>"
        elif signal == 'PUT':
            arrow_html = "<div style='font-size:28px;color:red'>⬇️</div>"
        else:
            arrow_html = "<div style='font-size:20px;color:gray'>—</div>"

        info_area.markdown(f"<div class='metric-card'><b>Sinal:</b> {signal} {arrow_html}</div>", unsafe_allow_html=True)

        # Plotly Candlestick with arrows annotation
        fig = go.Figure(data=[go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close']
        )])
        fig.update_layout(template='plotly_dark', height=600)

        # Add EMA traces
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA9'], name='EMA9', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA21'], name='EMA21', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA50'], name='EMA50', line=dict(width=1)))

        # Add arrow annotation for last signal
        last_time = df['time'].iloc[-1]
        last_close = df['close'].iloc[-1]
        if signal == 'CALL':
            fig.add_annotation(x=last_time, y=last_close, text='⬆', showarrow=True, arrowhead=3, arrowsize=2, arrowcolor='limegreen')
        elif signal == 'PUT':
            fig.add_annotation(x=last_time, y=last_close, text='⬇', showarrow=True, arrowhead=3, arrowsize=2, arrowcolor='red')

        chart_area.plotly_chart(fig, use_container_width=True)

        # Heatmap volatility
        heatmap_fig = go.Figure(data=go.Heatmap(z=[df['volatility'].tail(50)], colorscale='Inferno'))
        heatmap_fig.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # If we have a live signal and live mode on, schedule order
        if st.session_state['live'] and signal in ['CALL', 'PUT']:
            # compute amount
            if amount_fixed > 0:
                amount = amount_fixed
            else:
                # naive: assume we can compute amount from risk_percent and price - user can replace with balance-based calc
                price = df['close'].iloc[-1]
                # placeholder: amount equal to 0.001 * (risk_percent / 1)
                amount = 0.001 * (risk_percent / 1.0)

            # map signal to order function
            if exchange_choice == 'Binance':
                side = 'buy' if signal == 'CALL' else 'sell'
                # schedule order after 60 seconds
                delayed_order_executor('Binance', place_binance_order, 60, symbol, side, amount, {'apiKey': api_key, 'secret': api_secret}, test=test_mode)
                st.success(f'Order scheduled: {signal} in 60s (Binance)')
            elif exchange_choice == 'MT5':
                order_type = 'buy' if signal == 'CALL' else 'sell'
                delayed_order_executor('MT5', place_mt5_order, 60, symbol, order_type, amount)
                st.success(f'Order scheduled: {signal} in 60s (MT5)')
            elif exchange_choice == 'Deriv':
                st.warning('Deriv order logic is placeholder. Implement place_deriv_order and call delayed_order_executor')
            elif exchange_choice == 'IQ Option':
                st.warning('IQ Option order logic is placeholder. Implement place_iqoption_order and call delayed_order_executor')

    except Exception as e:
        st.exception(e)

# Main loop (Streamlit refresh driven)
if st.session_state['live']:
    update_and_render()
else:
    if st.button('Atualizar Agora'):
        update_and_render()

# Show last order result
if st.session_state.get('last_order_result') is not None:
    st.write('Último resultado da ordem:')
    st.write(st.session_state.get('last_order_result'))

# Footer notes
st.markdown('''
<style>
.metric-card{background:#111;padding:8px;border-radius:8px;margin-bottom:8px}
</style>
''', unsafe_allow_html=True)

st.info('Este template fornece integração Binance completa via ccxt e uma base para MT5. Deriv/ IQ Option precisam que você adicione seu conector (bibliotecas não-oficiais).')


# End of file

# --- IQ OPTION INTEGRAÇÃO COMPLETA ---
from iqoptionapi.stable_api import IQ_Option

# Config IQ Option
IQ_EMAIL = "SEU_EMAIL"
IQ_PASS = "SUA_SENHA"

iq = IQ_Option(IQ_EMAIL, IQ_PASS)
check, reason = iq.connect()

if check:
    print("✔ Conectado à IQ Option")
else:
    print("❌ Erro ao conectar:", reason)

# Função para enviar ordem

def enviar_ordem_iq(direcao, valor=1, par="EURUSD", duracao=1):
    if direcao == "CALL":
        iq.buy(valor, par, "call", duracao)
    elif direcao == "PUT":
        iq.buy(valor, par, "put", duracao)

# --- SINAL + CRONÔMETRO ---
import time

def executar_sinal_com_cronometro(signal, delay):
    if signal in ["CALL", "PUT"]:
        for t in range(delay, 0, -1):
            st.write(f"⏳ Entrada em {t}s...")
            time.sleep(1)
        st.success(f"🚀 Ordem enviada: {signal}")
        enviar_ordem_iq(signal)
    else:
        st.info("Nenhum sinal para enviar agora.")

# --- SETAS COLORIDAS NO GRÁFICO ---
import plotly.graph_objects as go

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["close"],
    mode="markers",
    marker=dict(
        size=14,
        color=["green" if s=="CALL" else "red" if s=="PUT" else "gray" for s in df["signal"]],
        symbol=["triangle-up" if s=="CALL" else "triangle-down" if s=="PUT" else "circle" for s in df["signal"]]
    ),
    name="Sinais"
))


# ======================= CONFIGURAÇÕES DO USUÁRIO =======================
cronometro = st.sidebar.number_input("⏳ Tempo do cronômetro (segundos)", min_value=1, max_value=300, value=60)
valor_entrada = st.sidebar.number_input("💰 Valor da entrada", min_value=1.0, value=1.0)
par_moeda = st.sidebar.text_input("💱 Par de moedas", value="EURUSD")
velas_operacao = st.sidebar.number_input("⏲️ Duração da operação (minutos)", min_value=1, max_value=15, value=1)
modo = st.sidebar.radio("🟢 Modo", ["Demo", "Real"])

# ======================= ENVIO DE ORDENS IQ OPTION =======================
from iqoptionapi.stable_api import IQ_Option

iq = IQ_Option(IQ_EMAIL, IQ_PASS)
check, reason = iq.connect()

if modo == "Demo":
    iq.change_balance("PRACTICE")
else:
    iq.change_balance("REAL")

# Função de envio
def enviar_ordem_iq(direcao):
    if direcao == "CALL":
        iq.buy(valor_entrada, par_moeda, "call", velas_operacao)
    elif direcao == "PUT":
        iq.buy(valor_entrada, par_moeda, "put", velas_operacao)

# ======================= CRONÔMETRO + ALARME =======================
import time

def executar_sinal(signal):
    if signal not in ["CALL", "PUT"]:
        st.info("Nenhum sinal válido no momento.")
        return

    for t in range(cronometro, 0, -1):
        st.write(f"⏳ Entrada em {t}s...")
        time.sleep(1)

    st.success(f"🚀 Ordem enviada: {signal}")

    # Alarme sonoro ao enviar o sinal
    st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg")

    enviar_ordem_iq(signal)


# ======================= TELEGRAM CONFIG =======================
import requests

tg_token = st.sidebar.text_input("🤖 Telegram Bot Token", value="", type="password")
tg_chat_id = st.sidebar.text_input("📨 Telegram Chat ID", value="")

# Função para enviar mensagem no Telegram
def enviar_telegram(msg):
    if tg_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        data = {"chat_id": tg_chat_id, "text": msg}
        try:
            requests.post(url, data=data)
        except:
            st.warning("⚠️ Não foi possível enviar a mensagem ao Telegram.")

# Envio automático do sinal

def enviar_sinal_telegram(signal):
    if signal in ["CALL", "PUT"]:
        enviar_telegram(f"📢 SINAL GERADO: {signal}")


# ======================= MENU LATERAL (PÁGINAS) =======================
pagina = st.sidebar.selectbox("📌 Navegação", ["Dashboard", "Configurações"])

# ======================= PÁGINA DE CONFIGURAÇÕES =======================
if pagina == "Configurações":
    st.title("⚙️ Configurações do Bot")

    st.subheader("⏳ Cronômetro")
    st.write(f"Atual: **{cronometro}s**")

    st.subheader("💰 Valor da Entrada")
    st.write(f"Atual: **{valor_entrada} USD**")

    st.subheader("💱 Par de Moedas")
    st.write(f"Atual: **{par_moeda}**")

    st.subheader("⏲️ Duração da Operação")
    st.write(f"Atual: **{velas_operacao} minutos**")

    st.subheader("🟢 Modo de Operação")
    st.write(f"Atual: **{modo}**")

    st.subheader("🤖 Telegram")
    st.write(f"Token definido: {'✅' if tg_token else '❌'}")
    st.write(f"Chat ID definido: {'✅' if tg_chat_id else '❌'}")

# ======================= RESULTADO WIN/LOSS PARA TELEGRAM =======================

def enviar_resultado_telegram(resultado, valor, par):
    msg = f"📊 *Resultado da Operação:*
💹 PAR: {par}
💵 VALOR: {valor}
🏆 RESULTADO: {resultado}"
    enviar_telegram(msg)

# Exemplo de uso após fechar operação:
# enviar_resultado_telegram("WIN", valor_entrada, par_moeda)


import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import io, base64
import ccxt
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# ===================== BANCO DE DADOS =====================
conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password_hash TEXT,
    tests_used INTEGER DEFAULT 0,
    created_at TEXT
)
""")
conn.commit()

# ===================== FUNÇÕES =====================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    cur.execute("SELECT id FROM users WHERE username=?", (username,))
    if cur.fetchone():
        return "username_exists"
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), datetime.now())
        )
        conn.commit()
        return "success"
    except:
        return "error"

def login_user(username, password):
    cur.execute("SELECT id, password_hash, tests_used FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    if user and user[1] == hash_password(password):
        return user
    return None

# ===================== BOT SIMPLES =====================
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
        """, unsafe_allow_html=True)
    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound(sound_up_b64)
        elif "DESCIDA" in signal:
            play_sound(sound_down_b64)

def fetch_crypto(symbol="BTC/USDT", exchange_name='binance', timeframe='15m', limit=300):
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def fetch_forex(symbol="EUR/USD", timeframe='15m', limit=300):
    date_rng = pd.date_range(end=datetime.now(), periods=limit, freq='15T')
    price = np.cumsum(np.random.randn(limit)*0.01) + 1.1
    df = pd.DataFrame({'ts': date_rng, 'open': price, 'high': price+0.005, 'low': price-0.005,
                       'close': price, 'volume': np.random.randint(100,500, size=limit)})
    df.set_index('ts', inplace=True)
    return df

def predict_signal(df):
    last_price = df['close'].iloc[-1]
    pred_price = last_price * (1 + np.random.uniform(-0.01,0.01))
    diff = (pred_price - last_price) / last_price
    confidence = np.random.uniform(60,98)
    signal = "SUBIDA 🔼" if diff > 0.002 else "DESCIDA 🔽" if diff < -0.002 else "NEUTRAL ⚪"
    return last_price, pred_price, diff, signal, confidence

# ===================== CSS =====================
st.markdown("""
<style>
@keyframes pulse {0% { box-shadow: 0 0 0 0 rgba(0,255,0,0.6);} 70% { box-shadow:0 0 20px 10px rgba(0,255,0,0);} 100% {box-shadow:0 0 0 0 rgba(0,255,0,0);}}
.pulse-green { animation: pulse 1.5s infinite; }
.pulse-red { animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Bot Trading PRO — Painel de IA")
st.sidebar.markdown("💰 Preço do Bot: **35 USDT**")

# ===================== LOGIN / REGISTRO =====================
if "user_id" not in st.session_state:
    auth_menu = st.selectbox("Escolha:", ["Login", "Criar Conta"])
    
    if auth_menu=="Criar Conta":
        st.subheader("📘 Criar Conta")
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        if st.button("Registrar"):
            result = register_user(username, password)
            if result=="success":
                st.success("Conta criada! Faça login.")
            elif result=="username_exists":
                st.error("Usuário já existe.")
            else:
                st.error("Erro inesperado ao criar conta.")
    
    elif auth_menu=="Login":
        st.subheader("🔐 Entrar")
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            user_data = login_user(username,password)
            if user_data:
                st.session_state["user_id"] = user_data[0]
                st.session_state["tests"] = user_data[2]
                st.success("Login realizado!")
            else:
                st.error("Usuário ou senha incorretos.")

else:
    st.success("✅ Login efetuado com sucesso!")

    # ----------------- BLOQUEIO DE TESTES -----------------
    max_tests = 2
    if st.session_state.get("tests",0) >= max_tests:
        st.error("❌ SEUS TESTES GRATUITOS ACABARAM")
        st.warning("⚠️ Antes de efetuar qualquer pagamento, entre em contato com o proprietário!")
        st.markdown("""
        ## 💳 Dados para Pagamento
        - Banco BFA: A06.0006.000.8110.483.2.3013.6  
        - Titular: ABRAAO HENJENGO PAULO  
        - Cripto USDT TRC20: TCqARvZ9VkRjJSfkRjJSfkHdi4XFgYpBfBQDG5Z  
        - WhatsApp: +244957691466  
        """)
        st.stop()

    # ----------------- MENU DE ESCOLHA -----------------
    st.sidebar.subheader("Escolha o mercado")
    market_choice = st.selectbox("Mercado:", ["Cripto (BTC/USDT)","Forex (EUR/USDT)"])

    # Mensagem incentivando PRO
    st.info("⚠️ Para liberar mais criptomoedas e pares Forex, adquira a versão PRO!")

    # ----------------- BOT -----------------
    if st.button("▶️ Analisar mercado"):
        # incrementa teste
        st.session_state["tests"] += 1
        cur.execute("UPDATE users SET tests_used=? WHERE id=?",
                    (st.session_state["tests"], st.session_state["user_id"]))
        conn.commit()

        symbol = "BTC/USDT" if "Cripto" in market_choice else "EUR/USD"
        df = fetch_crypto(symbol) if "Cripto" in market_choice else fetch_forex(symbol)
        last_price, pred_price, diff, signal, confidence = predict_signal(df)
        
        st.subheader(f"💰 Preço Atual: {last_price:.4f}")
        st.subheader(f"📈 Preço Previsto: {pred_price:.4f}")
        st.metric("Variação (%)", f"{diff*100:.2f}%")
        show_signal_alert(signal, confidence, 70)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Preço Real'))
        fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1]+pd.Timedelta(minutes=15)],
                                 y=[last_price, pred_price], mode='lines+markers', name='Previsão'))
        st.plotly_chart(fig,use_container_width=True)

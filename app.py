import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import io, base64
import ccxt
from datetime import datetime
import plotly.graph_objects as go

# Opcional: tenta importar RandomForest; se não existir, usa fallback simples
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ===================== BANCO DE DADOS (mantido) =====================
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

# ===================== FUNÇÕES DE AUTENTICAÇÃO (mantidas) =====================
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

# ===================== SONS / ALERTA =====================
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
        padding:1.1rem;border-radius:0.9rem;text-align:center;
        color:white;font-size:1.25rem;margin-bottom:8px;'>
        <b>{signal}</b><br>
        Confiança: {confidence:.1f}%
        </div>
        """, unsafe_allow_html=True)
    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound(sound_up_b64)
        elif "DESCIDA" in signal:
            play_sound(sound_down_b64)

# ===================== CRIPTO REAL (BINANCE) =====================
@st.cache_data(ttl=120)
def fetch_crypto(symbol="BTC/USDT", timeframe='15m', limit=300):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        # garantir markets carregadas para evitar erros intermitentes
        exchange.load_markets()
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        # garantir float
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df
    except Exception as e:
        # Não quebra o app; retorna None para o chamador lidar
        return None

# ===================== PREVISÃO (RandomForest se disponível, caso contrário momentum) =====================
def prepare_features(df):
    df2 = df.copy()
    df2['HL'] = df2['high'] - df2['low']
    df2['OC'] = df2['close'] - df2['open']
    df2['SMA5'] = df2['close'].rolling(5).mean()
    df2['SMA10'] = df2['close'].rolling(10).mean()
    df2['SMA_diff'] = df2['SMA5'] - df2['SMA10']
    df2.fillna(0, inplace=True)
    return df2

def predict_with_model(df):
    df_feat = prepare_features(df)
    # precisa de pelo menos 30 candles para treinar razoavelmente
    if SKLEARN_AVAILABLE and len(df_feat) >= 40:
        X = df_feat[['open','high','low','close','volume','HL','OC','SMA5','SMA10','SMA_diff']].iloc[:-1]
        y = df_feat['close'].shift(-1).iloc[:-1]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        next_X = X.iloc[-1:].values
        pred_price = model.predict(next_X)[0]
        last_price = float(df['close'].iloc[-1])
        diff_perc = ((pred_price - last_price) / last_price) * 100
        # sensibilidade aprimorada (pequenas variações agora têm impacto)
        if diff_perc > 0.05:
            signal = "SUBIDA 🔼"
        elif diff_perc < -0.05:
            signal = "DESCIDA 🔽"
        else:
            # usar momentum curto
            last2 = float(df['close'].iloc[-2])
            trend = ((last_price - last2) / last2) * 100
            signal = "SUBIDA 🔼" if trend > 0 else ("DESCIDA 🔽" if trend < 0 else "NEUTRAL ⚪")
        confidence = min(99, max(50, abs(diff_perc)*8 + 60))
        return last_price, pred_price, diff_perc, signal, confidence
    else:
        # fallback: momentum / extrapolação simples
        last_price = float(df['close'].iloc[-1])
        last2 = float(df['close'].iloc[-2])
        # extrapola preço pelo último movimento
        pred_price = last_price + (last_price - last2)
        diff_perc = ((pred_price - last_price) / last_price) * 100
        if diff_perc > 0.05:
            signal = "SUBIDA 🔼"
        elif diff_perc < -0.05:
            signal = "DESCIDA 🔽"
        else:
            signal = "SUBIDA 🔼" if (last_price - last2) > 0 else ("DESCIDA 🔽" if (last_price - last2) < 0 else "NEUTRAL ⚪")
        confidence = min(95, max(50, abs(diff_perc)*10 + 60))
        return last_price, pred_price, diff_perc, signal, confidence

# ===================== UI / PARES =====================
st.set_page_config(page_title="🚀 Bot Trading PRO — Cripto (Binance)", layout="wide")
st.markdown("<h2>🚀 Bot Trading PRO — Cripto (Binance) — Versão com múltiplos pares</h2>", unsafe_allow_html=True)

# lista dos pares que você pediu
PAIRS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
    "DOGE/USDT","ADA/USDT","AVAX/USDT","TRX/USDT","MATIC/USDT"
]

st.sidebar.markdown("💰 Preço do Bot: **35 USDT**")
st.sidebar.subheader("Escolha o par (Binance)")
pair = st.sidebar.selectbox("Par:", PAIRS, index=0)
timeframe = st.sidebar.selectbox("Timeframe:", ["1m","3m","5m","15m","30m","1h","4h","1d"], index=3)
confidence_threshold = st.sidebar.slider("🔉 Nível mínimo de confiança p/ alerta", 50, 100, 70, 1)

# containers para evitar problemas de DOM ao atualizar
chart_container = st.empty()
alert_container = st.empty()

# manter login/controle de testes como estava
if "user_id" not in st.session_state:
    auth_menu = st.selectbox("Escolha:", ["Login", "Criar Conta"])
    if auth_menu == "Criar Conta":
        st.subheader("📘 Criar Conta")
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        if st.button("Registrar"):
            result = register_user(username, password)
            if result == "success":
                st.success("Conta criada! Faça login.")
            elif result == "username_exists":
                st.error("Usuário já existe.")
            else:
                st.error("Erro inesperado ao criar conta.")
    else:
        st.subheader("🔐 Entrar")
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            user_data = login_user(username, password)
            if user_data:
                st.session_state["user_id"] = user_data[0]
                st.session_state["tests"] = user_data[2]
                st.success("Login realizado!")
            else:
                st.error("Usuário ou senha incorretos.")
else:
    st.success("✅ Login efetuado com sucesso!")

    # testes grátis
    max_tests = 2
    if st.session_state.get("tests", 0) >= max_tests:
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

    st.info("📌 Apenas pares reais da Binance (dados ao vivo).")

    if st.button("▶️ Analisar mercado"):
        # incrementa contador de testes
        st.session_state["tests"] = st.session_state.get("tests", 0) + 1
        cur.execute("UPDATE users SET tests_used=? WHERE id=?", (st.session_state["tests"], st.session_state["user_id"]))
        conn.commit()

        with st.spinner("🔍 Buscando dados reais da Binance..."):
            df = fetch_crypto(pair, timeframe=timeframe, limit=300)

        if df is None or df.empty:
            st.error("❌ Erro ao buscar dados reais. Tente outro par ou verifique sua conexão.")
        else:
            # previsão / análise
            last_price, pred_price, diff_perc, signal, confidence = predict_with_model(df)

            # mostrar resultados
            st.subheader(f"💰 Par: {pair}")
            st.subheader(f"💵 Preço Atual: {last_price:.6f}")
            st.subheader(f"📈 Preço Previsto: {pred_price:.6f}")
            st.metric("Variação (%)", f"{diff_perc:.4f}%")

            # alerta (usa alert_container)
            alert_container.empty()
            with st.container():
                show_signal_alert(signal, confidence, confidence_threshold)

            # gráfico (usa chart_container para evitar removeChild)
            chart_container.empty()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Preço Real'))
            # adiciona linha curta de projeção
            future_ts = df.index[-1] + (df.index[-1] - df.index[-2])
            fig.add_trace(go.Scatter(x=[df.index[-1], future_ts], y=[last_price, pred_price],
                                     mode='lines+markers', name='Previsão'))
            fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            chart_container.plotly_chart(fig, use_container_width=True)


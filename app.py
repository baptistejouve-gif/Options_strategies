import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Options Strategy Builder", layout="wide")

st.title("Options Strategy Builder")
st.write("Construis librement une stratégie d'options. "
         "La prime est calculée automatiquement avec Black-Scholes.")

# --- FONCTIONS BLACK-SCHOLES ---
def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(S - K, 0) if option_type == "Call" else max(K - S, 0)

    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    if option_type == "Call":
        return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

def bs_greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}

    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    gamma = norm.pdf(d_1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d_1) * np.sqrt(T) / 100

    if option_type == "Call":
        delta = norm.cdf(d_1)
        theta = ( -S * norm.pdf(d_1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d_2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d_2) / 100
    else:
        delta = norm.cdf(d_1) - 1
        theta = (-S * norm.pdf(d_1) * sigma / (2 * np.sqrt(T))+ r * K * np.exp(-r * T) * norm.cdf(-d_2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d_2) / 100

    return {"Delta": delta,"Gamma": gamma,"Vega": vega, "Theta": theta, "Rho": rho}

# --- FONCTIONS PAYOFF ET P&L ---
def option_payoff(ST, K, option_type, position, quantity):
    if option_type == "Call":
        intrinsic = np.maximum(ST - K, 0)
    else:
        intrinsic = np.maximum(K - ST, 0)

    payoff = intrinsic if position == "Long" else -intrinsic
    return quantity * payoff

def option_pnl(ST, K, premium, option_type, position, quantity):
    payoff = option_payoff(ST, K, option_type, position, quantity)

    if position == "Long":
        premium_cashflow = -premium * quantity
    else:
        premium_cashflow = premium * quantity

    return payoff + premium_cashflow

def sign_position(position):
    return 1 if position == "Long" else -1

# --- DÉTECTION AUTOMATIQUE DE STRATÉGIE ---
def detect_strategy(options):
    n = len(options)

    if n == 1:
        o = options[0]
        return f"{o['Position']} {o['Type']}"

    calls = [o for o in options if o["Type"] == "Call"]
    puts = [o for o in options if o["Type"] == "Put"]
    longs = [o for o in options if o["Position"] == "Long"]
    shorts = [o for o in options if o["Position"] == "Short"]

    strikes = sorted([o["Strike"] for o in options])
    maturities = sorted([o["Maturité"] for o in options])
    same_maturity = len(set(maturities)) == 1
    same_strike = len(set(strikes)) == 1

    if n == 2 and len(calls) == 1 and len(puts) == 1 and same_strike and same_maturity:
        if len(longs) == 2:
            return "Long Straddle"
        if len(shorts) == 2:
            return "Short Straddle"

    if n == 2 and len(calls) == 1 and len(puts) == 1 and not same_strike and same_maturity:
        call = calls[0]
        put = puts[0]

        if put["Strike"] < call["Strike"]:
            if len(longs) == 2:
                return "Long Strangle"
            if len(shorts) == 2:
                return "Short Strangle"

        if call["Strike"] < put["Strike"]:
            if len(longs) == 2:
                return "Long Guts"
            if len(shorts) == 2:
                return "Short Guts"

    if n == 2 and len(calls) == 1 and len(puts) == 1:
        call = calls[0]
        put = puts[0]

        if call["Strike"] == put["Strike"] and call["Position"] == "Long" and put["Position"] == "Short":
            return "Synthetic Long Forward"

        if call["Strike"] == put["Strike"] and call["Position"] == "Short" and put["Position"] == "Long":
            return "Synthetic Short Forward"

        if call["Position"] == "Long" and put["Position"] == "Short":
            return "Bullish Risk Reversal"

        if call["Position"] == "Short" and put["Position"] == "Long":
            return "Bearish Risk Reversal"

    if n == 2 and len(calls) == 2 and same_maturity:
        call_long = [o for o in calls if o["Position"] == "Long"]
        call_short = [o for o in calls if o["Position"] == "Short"]

        if len(call_long) == 1 and len(call_short) == 1:
            K_long = call_long[0]["Strike"]
            K_short = call_short[0]["Strike"]

            if K_long < K_short:
                return "Bull Call Spread"
            if K_long > K_short:
                return "Bear Call Spread"

    if n == 2 and len(puts) == 2 and same_maturity:
        put_long = [o for o in puts if o["Position"] == "Long"]
        put_short = [o for o in puts if o["Position"] == "Short"]

        if len(put_long) == 1 and len(put_short) == 1:
            K_long = put_long[0]["Strike"]
            K_short = put_short[0]["Strike"]

            if K_long > K_short:
                return "Bear Put Spread"
            if K_long < K_short:
                return "Bull Put Spread"

    if n == 2 and len(set(strikes)) == 1 and len(set(maturities)) == 2:
        if len(calls) == 2:
            long_leg = [o for o in calls if o["Position"] == "Long"][0]
            short_leg = [o for o in calls if o["Position"] == "Short"][0]

            if long_leg["Maturité"] > short_leg["Maturité"]:
                return "Long Call Calendar Spread"
            else:
                return "Short Call Calendar Spread"

        if len(puts) == 2:
            long_leg = [o for o in puts if o["Position"] == "Long"][0]
            short_leg = [o for o in puts if o["Position"] == "Short"][0]

            if long_leg["Maturité"] > short_leg["Maturité"]:
                return "Long Put Calendar Spread"
            else:
                return "Short Put Calendar Spread"

    if n == 2 and len(set(strikes)) == 2 and len(set(maturities)) == 2:
        if len(calls) == 2:
            return "Call Diagonal Spread"
        if len(puts) == 2:
            return "Put Diagonal Spread"

    if n == 3 and same_maturity:
        if len(calls) == 3:
            qty = sorted([o["Quantité"] for o in calls])
            if qty == [1, 1, 2] and len(set(strikes)) == 3:
                middle_strike = strikes[1]
                middle_leg = [o for o in calls if o["Strike"] == middle_strike][0]

                if middle_leg["Position"] == "Short":
                    return "Long Call Butterfly"
                if middle_leg["Position"] == "Long":
                    return "Short Call Butterfly"

        if len(puts) == 3:
            qty = sorted([o["Quantité"] for o in puts])
            if qty == [1, 1, 2] and len(set(strikes)) == 3:
                middle_strike = strikes[1]
                middle_leg = [o for o in puts if o["Strike"] == middle_strike][0]

                if middle_leg["Position"] == "Short":
                    return "Long Put Butterfly"
                if middle_leg["Position"] == "Long":
                    return "Short Put Butterfly"

    if n == 4 and len(calls) == 2 and len(puts) == 2 and same_maturity:
        short_call = [o for o in calls if o["Position"] == "Short"]
        short_put = [o for o in puts if o["Position"] == "Short"]
        long_call = [o for o in calls if o["Position"] == "Long"]
        long_put = [o for o in puts if o["Position"] == "Long"]

        if len(short_call) == 1 and len(short_put) == 1 and len(long_call) == 1 and len(long_put) == 1:
            if short_call[0]["Strike"] == short_put[0]["Strike"]:
                return "Iron Butterfly"

            if long_call[0]["Strike"] == long_put[0]["Strike"]:
                return "Reverse Iron Butterfly"

    if n == 4 and len(calls) == 2 and len(puts) == 2 and same_maturity:
        call_long = [o for o in calls if o["Position"] == "Long"]
        call_short = [o for o in calls if o["Position"] == "Short"]
        put_long = [o for o in puts if o["Position"] == "Long"]
        put_short = [o for o in puts if o["Position"] == "Short"]

        if len(call_long) == 1 and len(call_short) == 1 and len(put_long) == 1 and len(put_short) == 1:
            K_put_long = put_long[0]["Strike"]
            K_put_short = put_short[0]["Strike"]
            K_call_short = call_short[0]["Strike"]
            K_call_long = call_long[0]["Strike"]

            if K_put_long < K_put_short < K_call_short < K_call_long:
                return "Iron Condor"

            if K_put_short < K_put_long < K_call_long < K_call_short:
                return "Reverse Iron Condor"

    return "Stratégie personnalisée / non reconnue"

# --- INTERPRÉTATION DES GREEKS ---
def greek_exposure_comment(name, value):
    if abs(value) < 1e-6:
        return f"Neutral {name.lower()}"
    return f"Long {name.lower()}" if value > 0 else f"Short {name.lower()}"

# --- PARAMÈTRES UTILISATEUR ---
st.sidebar.header("Paramètres globaux")

spot = st.sidebar.number_input("Spot actuel", min_value=1.0, value=100.0, step=1.0)

risk_free_rate = st.sidebar.number_input("Taux sans risque r", min_value=-0.10, max_value=0.30, value=0.03,step=0.005, format="%.3f")

price_min = st.sidebar.number_input("Prix min affiché", min_value=0.0, value=50.0, step=5.0)
price_max = st.sidebar.number_input("Prix max affiché", min_value=1.0, value=150.0, step=5.0)

selected_ST = st.sidebar.slider( "Prix du sous-jacent à maturité", min_value=float(price_min), max_value=float(price_max), value=float(spot),step=1.0)

number_options = st.sidebar.number_input("Nombre d'options",min_value=1,max_value=12,value=2,step=1)

# --- CONSTRUCTION DES OPTIONS ---
st.markdown("### Construction de la stratégie")

options = []

for i in range(int(number_options)):
    st.markdown(f"#### Option {i + 1}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        option_type = st.selectbox("Type", ["Call", "Put"], key=f"type_{i}")

    with col2:
        position = st.selectbox("Position", ["Long", "Short"], key=f"position_{i}")

    with col3:
        strike = st.number_input("Strike", min_value=1.0, value=100.0, step=1.0, key=f"strike_{i}")

    with col4:
        quantity = st.number_input("Quantité", min_value=1, value=1, step=1, key=f"quantity_{i}")

    with col5:
        maturity = st.number_input("Maturité T", min_value=0.001, value=0.25, step=0.05, key=f"maturity_{i}")

    with col6:
        implied_vol = st.number_input("Vol implicite", min_value=0.001, value=0.20, step=0.01, key=f"vol_{i}")

    premium = bs_price(spot,strike,maturity,risk_free_rate,implied_vol,option_type)

    st.caption(f"Prime théorique Black-Scholes : {premium:.4f}")

    greeks = bs_greeks(spot,strike,maturity,risk_free_rate, implied_vol,option_type)

    multiplier = sign_position(position) * quantity

    options.append({"Type": option_type,"Position": position,"Strike": strike,"Prime": premium,"Quantité": quantity,"Maturité": maturity,"Vol": implied_vol,"Prix BS": premium,"Delta": greeks["Delta"] * multiplier,"Gamma": greeks["Gamma"] * multiplier,"Vega": greeks["Vega"] * multiplier,"Theta": greeks["Theta"] * multiplier,"Rho": greeks["Rho"] * multiplier})

# --- CALCUL PAYOFF, P&L ET GREEKS ---
ST_range = np.linspace(price_min, price_max, 700)

total_payoff = np.zeros_like(ST_range)
total_pnl = np.zeros_like(ST_range)

for opt in options:
    total_payoff += option_payoff(ST_range,opt["Strike"],opt["Type"],opt["Position"],opt["Quantité"])

    total_pnl += option_pnl(ST_range,opt["Strike"],opt["Prime"],opt["Type"],opt["Position"],opt["Quantité"])

selected_payoff = np.interp(selected_ST, ST_range, total_payoff)
selected_pnl = np.interp(selected_ST, ST_range, total_pnl)

strategy_detected = detect_strategy(options)

total_delta = sum(o["Delta"] for o in options)
total_gamma = sum(o["Gamma"] for o in options)
total_vega = sum(o["Vega"] for o in options)
total_theta = sum(o["Theta"] for o in options)
total_rho = sum(o["Rho"] for o in options)

net_premium = sum(o["Prime"] * o["Quantité"] * (-1 if o["Position"] == "Long" else 1)for o in options)

breakeven_indices = np.where(np.diff(np.sign(total_pnl)) != 0)[0]
breakevens = []

for idx in breakeven_indices:
    x1, x2 = ST_range[idx], ST_range[idx + 1]
    y1, y2 = total_pnl[idx], total_pnl[idx + 1]

    if y2 != y1:
        breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
        breakevens.append(breakeven)

# --- AFFICHAGE RÉSUMÉ ---
st.markdown("### Résumé de la stratégie")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Stratégie détectée", strategy_detected)
col2.metric("P&L au prix choisi", f"{selected_pnl:.2f}")
col3.metric("Cash-flow prime", f"{net_premium:.2f}")
col4.metric("Prix final choisi", f"{selected_ST:.2f}")

col5, col6, col7 = st.columns(3)

col5.metric("Gain max zone affichée", f"{np.max(total_pnl):.2f}")
col6.metric("Perte max zone affichée", f"{np.min(total_pnl):.2f}")

col7.metric("Breakeven(s)",", ".join([f"{x:.2f}" for x in breakevens]) if breakevens else "Aucun")

# --- GRAPHIQUE INTERACTIF PAYOFF / P&L ---
st.markdown("### Payoff / P&L à maturité")

display_mode = st.radio("Affichage",["P&L net", "Payoff brut", "Les deux"],horizontal=True)

fig_payoff = go.Figure()

if display_mode in ["P&L net", "Les deux"]:
    pnl_positive = np.where(total_pnl >= 0, total_pnl, np.nan)
    pnl_negative = np.where(total_pnl < 0, total_pnl, np.nan)

    fig_payoff.add_trace(go.Scatter(x=ST_range,y=pnl_positive,mode="lines",name="P&L positif",line=dict(color="green")))

    fig_payoff.add_trace(go.Scatter(x=ST_range,y=pnl_negative,mode="lines",name="P&L négatif",line=dict(color="red")))

if display_mode in ["Payoff brut", "Les deux"]:
    fig_payoff.add_trace(go.Scatter(x=ST_range,y=total_payoff,mode="lines",name="Payoff brut",line=dict(color="blue", dash="dash")))

fig_payoff.add_hline(y=0,line_dash="dash",annotation_text="P&L = 0")

for be in breakevens:
    fig_payoff.add_vline(x=be,line_dash="dash",annotation_text=f"BE = {be:.2f}")

fig_payoff.add_vline(x=selected_ST,line_dash="dot",annotation_text=f"S_T = {selected_ST:.2f}")

fig_payoff.update_layout(title="Payoff brut vs P&L net de la stratégie",xaxis_title="Prix du sous-jacent à maturité",yaxis_title="Gain / Perte",hovermode="x unified",height=600)

st.plotly_chart(fig_payoff, use_container_width=True)

# --- AFFICHAGE DES GREEKS ---
st.markdown("### Exposition aux Greeks")

g1, g2, g3, g4, g5 = st.columns(5)

g1.metric("Delta", f"{total_delta:.4f}")
g2.metric("Gamma", f"{total_gamma:.4f}")
g3.metric("Vega", f"{total_vega:.4f}")
g4.metric("Theta", f"{total_theta:.4f}")
g5.metric("Rho", f"{total_rho:.4f}")

st.markdown("### Interprétation")

st.write(greek_exposure_comment("Delta", total_delta))
st.write(greek_exposure_comment("Gamma", total_gamma))
st.write(greek_exposure_comment("Vega", total_vega))
st.write(greek_exposure_comment("Theta", total_theta))
st.write(greek_exposure_comment("Rho", total_rho))

if total_gamma > 0 and total_vega > 0 and total_theta < 0:
    st.info("Profil long convexité : long gamma, long vega, short theta.")
elif total_gamma < 0 and total_vega < 0 and total_theta > 0:
    st.info("Profil vendeur de volatilité : short gamma, short vega, long theta.")
elif total_delta > 0:
    st.info("Profil directionnel haussier.")
elif total_delta < 0:
    st.info("Profil directionnel baissier.")
else:
    st.info("Profil globalement delta-neutre autour du spot actuel.")

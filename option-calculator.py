# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import date


st.set_page_config(page_title="Options Chain CAGR", layout="wide")

st.title("📊 Options Chain - Calls | Strike | Puts")
st.write("CAGR is calculated using LAST option price only.")


# =====================================================
# Session state defaults
# =====================================================
if "option_chain_loaded" not in st.session_state:
    st.session_state.option_chain_loaded = False

if "option_table" not in st.session_state:
    st.session_state.option_table = None

if "option_stock_price" not in st.session_state:
    st.session_state.option_stock_price = None

if "option_expiration" not in st.session_state:
    st.session_state.option_expiration = None

if "option_dte" not in st.session_state:
    st.session_state.option_dte = None

if "option_ticker" not in st.session_state:
    st.session_state.option_ticker = None

if "raw_calls" not in st.session_state:
    st.session_state.raw_calls = None

if "raw_puts" not in st.session_state:
    st.session_state.raw_puts = None

if "put_ladder_base_df" not in st.session_state:
    st.session_state.put_ladder_base_df = None

if "put_ladder_params" not in st.session_state:
    st.session_state.put_ladder_params = None


# =====================================================
# Helper functions
# =====================================================
def get_stock_price(ticker_obj):
    try:
        price = ticker_obj.fast_info.get("last_price")
        if price is not None and price > 0:
            return float(price)
    except Exception:
        pass

    hist = ticker_obj.history(period="5d")
    if hist.empty:
        raise ValueError("Could not get stock price.")

    return float(hist["Close"].dropna().iloc[-1])


def bs_delta(option_type, S, K, T, r, iv):
    if pd.isna(iv) or iv <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))

    if option_type == "CALL":
        return norm.cdf(d1)

    return norm.cdf(d1) - 1


def calc_cagr_by_last_price(last_price, capital_base, dte):
    """
    CAGR = (1 + last_price / capital_base) ^ (365 / DTE) - 1
    Returned as percent.
    """
    if pd.isna(last_price) or pd.isna(capital_base):
        return np.nan

    if last_price <= 0 or capital_base <= 0 or dte <= 0:
        return np.nan

    return ((1 + last_price / capital_base) ** (365 / dte) - 1) * 100


def prepare_calls(calls, stock_price, dte, r):
    calls = calls.copy()
    T = dte / 365

    calls["Call IV %"] = calls["impliedVolatility"] * 100

    calls["Call Delta"] = calls.apply(
        lambda row: bs_delta(
            option_type="CALL",
            S=stock_price,
            K=row["strike"],
            T=T,
            r=r,
            iv=row["impliedVolatility"]
        ),
        axis=1
    )

    calls["Call ITM"] = stock_price > calls["strike"]

    # Call period return by LAST CALL PRICE
    # Capital base = stock price
    calls["Call Period Return %"] = calls.apply(
        lambda row: (row["lastPrice"] / stock_price) * 100
        if not pd.isna(row["lastPrice"]) and row["lastPrice"] >= 0 and stock_price > 0
        else np.nan,
        axis=1
    )

    # Call CAGR by LAST CALL PRICE
    # Capital base = stock price, like covered call logic
    calls["Call CAGR %"] = calls.apply(
        lambda row: calc_cagr_by_last_price(
            last_price=row["lastPrice"],
            capital_base=stock_price,
            dte=dte
        ),
        axis=1
    )

    calls = calls.rename(columns={
        "lastPrice": "Call Last",
        "bid": "Call Bid",
        "ask": "Call Ask",
        "volume": "Call Volume",
        "openInterest": "Call OI"
    })

    keep_cols = [
        "strike",
        "Call Last",
        "Call Bid",
        "Call Ask",
        "Call Volume",
        "Call OI",
        "Call IV %",
        "Call Delta",
        "Call Period Return %",
        "Call CAGR %",
        "Call ITM"
    ]

    return calls[keep_cols]


def prepare_puts(puts, stock_price, dte, r):
    puts = puts.copy()
    T = dte / 365

    puts["Put IV %"] = puts["impliedVolatility"] * 100

    puts["Put Delta"] = puts.apply(
        lambda row: bs_delta(
            option_type="PUT",
            S=stock_price,
            K=row["strike"],
            T=T,
            r=r,
            iv=row["impliedVolatility"]
        ),
        axis=1
    )

    puts["Put ITM"] = stock_price < puts["strike"]

    # Put period return by LAST PUT PRICE
    # Period Return = Put Last Price / Strike
    puts["Put Period Return %"] = puts.apply(
        lambda row: (row["lastPrice"] / row["strike"]) * 100
        if not pd.isna(row["lastPrice"]) and row["lastPrice"] >= 0 and row["strike"] > 0
        else np.nan,
        axis=1
    )

    # Put CAGR by LAST PUT PRICE
    # Capital base = strike, like cash-secured put logic
    puts["Put CAGR %"] = puts.apply(
        lambda row: calc_cagr_by_last_price(
            last_price=row["lastPrice"],
            capital_base=row["strike"],
            dte=dte
        ),
        axis=1
    )

    puts = puts.rename(columns={
        "lastPrice": "Put Last",
        "bid": "Put Bid",
        "ask": "Put Ask",
        "volume": "Put Volume",
        "openInterest": "Put OI"
    })

    keep_cols = [
        "strike",
        "Put Last",
        "Put Bid",
        "Put Ask",
        "Put Volume",
        "Put OI",
        "Put IV %",
        "Put Delta",
        "Put Period Return %",
        "Put CAGR %",
        "Put ITM"
    ]

    return puts[keep_cols]


def style_itm(row):
    styles = [""] * len(row)

    call_itm = bool(row.get("Call ITM", False)) if not pd.isna(row.get("Call ITM", np.nan)) else False
    put_itm = bool(row.get("Put ITM", False)) if not pd.isna(row.get("Put ITM", np.nan)) else False

    for i, col in enumerate(row.index):
        if col.startswith("Call") and call_itm:
            styles[i] = "background-color: #264b70; color: white;"

        if col.startswith("Put") and put_itm:
            styles[i] = "background-color: #264b70; color: white;"

        if col == "Strike":
            styles[i] = "background-color: #333333; color: white; font-weight: bold;"

    return styles


def build_put_ladder_df(ladder_rows, total_capital, dte):
    """
    Build ladder dataframe from manual rows.

    Required columns in ladder_rows:
    Leg, Allocation %, Strike, Premium
    """
    ladder_df = pd.DataFrame(ladder_rows).copy()

    ladder_df["Allocation Weight"] = ladder_df["Allocation %"] / 100
    ladder_df["Target Capital"] = total_capital * ladder_df["Allocation Weight"]

    # Cash-secured put collateral
    ladder_df["Collateral Per Contract"] = ladder_df["Strike"] * 100

    # Contracts per strike
    ladder_df["Contracts"] = np.floor(
        ladder_df["Target Capital"] / ladder_df["Collateral Per Contract"]
    ).astype(int)

    return recalc_ladder_with_contracts(
        ladder_df=ladder_df,
        total_capital=total_capital,
        dte=dte,
        contracts_col="Contracts"
    )


def recalc_ladder_with_contracts(ladder_df, total_capital, dte, contracts_col="Contracts"):
    """
    Recalculate all ladder metrics using a selected contracts column.
    """
    df = ladder_df.copy()

    df[contracts_col] = df[contracts_col].fillna(0).astype(int)
    df[contracts_col] = df[contracts_col].clip(lower=0)

    # Rename the selected contract column into Final Contracts for consistent calculations
    if contracts_col != "Final Contracts":
        df["Final Contracts"] = df[contracts_col]
    else:
        df["Final Contracts"] = df[contracts_col]

    df["Actual Collateral"] = (
        df["Final Contracts"] * df["Strike"] * 100
    )

    df["Actual Weight %"] = np.where(
        total_capital > 0,
        df["Actual Collateral"] / total_capital * 100,
        np.nan
    )

    df["Unused Capital By Target"] = (
        df["Target Capital"] - df["Actual Collateral"]
    )

    df["Premium Cash"] = (
        df["Final Contracts"] * df["Premium"] * 100
    )

    df["Period Return %"] = np.where(
        df["Actual Collateral"] > 0,
        df["Premium Cash"] / df["Actual Collateral"] * 100,
        np.nan
    )

    df["Simple Annualized %"] = df["Period Return %"] * 365 / dte

    df["CAGR %"] = np.where(
        df["Period Return %"].notna(),
        ((1 + df["Period Return %"] / 100) ** (365 / dte) - 1) * 100,
        np.nan
    )

    df["Net Assignment Price"] = df["Strike"] - df["Premium"]
    df["Shares If Assigned"] = df["Final Contracts"] * 100

    return df


def summarize_ladder(ladder_df, total_capital, dte):
    """
    Return basket-level summary metrics.
    """
    total_actual_collateral = ladder_df["Actual Collateral"].sum()
    total_premium_cash = ladder_df["Premium Cash"].sum()
    total_unused_capital = total_capital - total_actual_collateral
    total_contracts = ladder_df["Final Contracts"].sum()
    total_shares_if_assigned = ladder_df["Shares If Assigned"].sum()

    if total_actual_collateral > 0:
        weighted_period_return = total_premium_cash / total_actual_collateral
        weighted_simple_annualized = weighted_period_return * 365 / dte
        weighted_cagr = ((1 + weighted_period_return) ** (365 / dte) - 1)
    else:
        weighted_period_return = np.nan
        weighted_simple_annualized = np.nan
        weighted_cagr = np.nan

    # NEW LOGIC:
    # Avg Assignment Price based only on Strike and Actual Weight %
    valid_weights = (
        ladder_df["Actual Weight %"].notna() &
        (ladder_df["Actual Weight %"] > 0)
    )

    if valid_weights.any():
        weight_sum = ladder_df.loc[valid_weights, "Actual Weight %"].sum()

        avg_assignment_price = (
            ladder_df.loc[valid_weights, "Strike"] *
            ladder_df.loc[valid_weights, "Actual Weight %"]
        ).sum() / weight_sum
    else:
        avg_assignment_price = np.nan

    return {
        "total_actual_collateral": total_actual_collateral,
        "total_premium_cash": total_premium_cash,
        "total_unused_capital": total_unused_capital,
        "total_contracts": total_contracts,
        "total_shares_if_assigned": total_shares_if_assigned,
        "weighted_period_return": weighted_period_return,
        "weighted_simple_annualized": weighted_simple_annualized,
        "weighted_cagr": weighted_cagr,
        "avg_assignment_price": avg_assignment_price
    }


def display_ladder_summary(summary, title="Ladder Summary"):
    st.markdown(f"### {title}")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Contracts", f"{summary['total_contracts']:,}")
    s2.metric("Total Premium", f"${summary['total_premium_cash']:,.2f}")
    s3.metric("Used Collateral", f"${summary['total_actual_collateral']:,.2f}")
    s4.metric("Unused Capital", f"${summary['total_unused_capital']:,.2f}")

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Weighted Period Return", f"{summary['weighted_period_return'] * 100:.2f}%")
    s6.metric("Weighted Simple Annualized", f"{summary['weighted_simple_annualized'] * 100:.2f}%")
    s7.metric("Weighted CAGR", f"{summary['weighted_cagr'] * 100:.2f}%")
    s8.metric("Avg Assignment Price", f"${summary['avg_assignment_price']:,.2f}")


def display_ladder_details(ladder_df, title="Ladder Details", download_key="ladder_csv"):
    st.markdown(f"### {title}")

    display_cols = [
        "Leg",
        "Allocation %",
        "Strike",
        "Premium",
        "Target Capital",
        "Contracts",
        "Extra Contracts",
        "Final Contracts",
        "Actual Collateral",
        "Actual Weight %",
        "Unused Capital By Target",
        "Premium Cash",
        "Period Return %",
        "Simple Annualized %",
        "CAGR %",
        "Net Assignment Price",
        "Shares If Assigned"
    ]

    existing_cols = [col for col in display_cols if col in ladder_df.columns]
    ladder_display = ladder_df[existing_cols].copy()

    for col in ladder_display.columns:
        if pd.api.types.is_numeric_dtype(ladder_display[col]):
            ladder_display[col] = ladder_display[col].round(2)

    st.dataframe(
        ladder_display,
        use_container_width=True,
        height=350
    )

    ladder_csv = ladder_display.to_csv(index=False).encode("utf-8")

    # IMPORTANT:
    # This key must be unique because this function is called twice:
    # 1. Base Ladder Details
    # 2. Final Ladder Details After Extra Contracts
    st.download_button(
        "Download Ladder CSV",
        data=ladder_csv,
        file_name=f"{download_key}.csv",
        mime="text/csv",
        key=f"download_{download_key}"
    )


# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

risk_free_rate_pct = st.sidebar.number_input(
    "Risk-free rate for delta %",
    value=4.0,
    step=0.1
)

risk_free_rate = risk_free_rate_pct / 100

filter_near_money = st.sidebar.checkbox(
    "Show only strikes near stock price",
    value=True
)

moneyness_range = st.sidebar.slider(
    "Strike range +/- %",
    min_value=5,
    max_value=100,
    value=100,
    step=5
)

show_raw_data = st.sidebar.checkbox("Show raw calls and puts", value=False)


# =====================================================
# Main app
# =====================================================
if ticker:
    try:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options)

        if len(expirations) == 0:
            st.error("No options found for this ticker.")
            st.stop()

        expiration = st.sidebar.selectbox("Expiration", expirations)

        load_button = st.sidebar.button("Load option chain")

        if load_button:
            stock_price = get_stock_price(tk)

            exp_date = pd.to_datetime(expiration).date()
            dte = max((exp_date - date.today()).days, 1)

            chain = tk.option_chain(expiration)

            calls = prepare_calls(
                calls=chain.calls,
                stock_price=stock_price,
                dte=dte,
                r=risk_free_rate
            )

            puts = prepare_puts(
                puts=chain.puts,
                stock_price=stock_price,
                dte=dte,
                r=risk_free_rate
            )

            table = pd.merge(
                calls,
                puts,
                on="strike",
                how="outer"
            ).sort_values("strike")

            table = table.rename(columns={"strike": "Strike"})

            if filter_near_money:
                lower = stock_price * (1 - moneyness_range / 100)
                upper = stock_price * (1 + moneyness_range / 100)
                table = table[
                    (table["Strike"] >= lower) &
                    (table["Strike"] <= upper)
                ]

            # Put Strike in the middle, like Yahoo straddle view
            cols = [
                "Call Last",
                "Call Bid",
                "Call Ask",
                "Call Volume",
                "Call OI",
                "Call IV %",
                "Call Delta",
                "Call Period Return %",
                "Call CAGR %",
                "Call ITM",

                "Strike",

                "Put Last",
                "Put Bid",
                "Put Ask",
                "Put Volume",
                "Put OI",
                "Put IV %",
                "Put Delta",
                "Put Period Return %",
                "Put CAGR %",
                "Put ITM"
            ]

            table = table[cols]

            # Round numeric columns
            for col in table.columns:
                if pd.api.types.is_numeric_dtype(table[col]):
                    if "Volume" in col or "OI" in col:
                        table[col] = table[col].fillna(0).astype(int)
                    else:
                        table[col] = table[col].round(2)

            # Save everything in session_state
            st.session_state.option_chain_loaded = True
            st.session_state.option_table = table
            st.session_state.option_stock_price = stock_price
            st.session_state.option_expiration = expiration
            st.session_state.option_dte = dte
            st.session_state.option_ticker = ticker
            st.session_state.raw_calls = chain.calls
            st.session_state.raw_puts = chain.puts

        # Display saved option chain even after calculator clicks
        if st.session_state.option_chain_loaded and st.session_state.option_table is not None:
            table = st.session_state.option_table
            stock_price = st.session_state.option_stock_price
            expiration_loaded = st.session_state.option_expiration
            dte_loaded = st.session_state.option_dte
            ticker_loaded = st.session_state.option_ticker

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ticker", ticker_loaded)
            c2.metric("Stock Price", f"{stock_price:,.2f}")
            c3.metric("Expiration", expiration_loaded)
            c4.metric("DTE", dte_loaded)

            st.warning(
                "CAGR here uses lastPrice only. LastPrice can be stale. "
                "For ITM options, lastPrice includes intrinsic value, so CAGR can be misleading."
            )

            st.subheader("Yahoo Style Option Chain")

            styled_table = table.style.apply(style_itm, axis=1)

            st.dataframe(
                styled_table,
                use_container_width=True,
                height=800
            )

            csv = table.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{ticker_loaded}_{expiration_loaded}_options_chain.csv",
                mime="text/csv"
            )

            if show_raw_data:
                st.subheader("Raw Calls")
                st.dataframe(st.session_state.raw_calls, use_container_width=True)

                st.subheader("Raw Puts")
                st.dataframe(st.session_state.raw_puts, use_container_width=True)

            st.markdown(
                """
                ### Table formulas used

                **Put Period Return %**

                `Put Period Return % = Put Last Price / Strike * 100`

                **Put CAGR by last put price**

                `Put CAGR % = ((1 + Put Last Price / Strike) ** (365 / DTE) - 1) * 100`

                **Call Period Return %**

                `Call Period Return % = Call Last Price / Stock Price * 100`

                **Call CAGR by last call price**

                `Call CAGR % = ((1 + Call Last Price / Stock Price) ** (365 / DTE) - 1) * 100`
                """
            )

    except Exception as e:
        st.error(f"Error: {e}")


# =====================================================
# Manual CAGR Calculator
# =====================================================
st.divider()
st.subheader("🧮 Manual CAGR Calculator")

st.write(
    "Enter the numbers manually and click Calculate. "
    "Formula: CAGR = (1 + premium / strike) ** (365 / DTE) - 1"
)

with st.form("manual_cagr_form"):
    calc_col1, calc_col2, calc_col3 = st.columns(3)

    manual_strike = calc_col1.number_input(
        "Strike / Capital Base",
        min_value=0.0,
        value=None,
        step=0.01,
        placeholder="Example: 50"
    )

    manual_premium = calc_col2.number_input(
        "Premium",
        min_value=0.0,
        value=None,
        step=0.01,
        placeholder="Example: 2"
    )

    manual_dte = calc_col3.number_input(
        "DTE",
        min_value=1,
        value=None,
        step=1,
        placeholder="Example: 84"
    )

    calculate_button = st.form_submit_button("Calculate CAGR")

if calculate_button:
    if manual_strike is None or manual_premium is None or manual_dte is None:
        st.error("Please enter Strike, Premium and DTE.")
    elif manual_strike <= 0:
        st.error("Strike / Capital Base must be greater than 0.")
    elif manual_dte <= 0:
        st.error("DTE must be greater than 0.")
    else:
        period_return = manual_premium / manual_strike

        simple_annualized = period_return * (365 / manual_dte) * 100

        manual_cagr = ((1 + period_return) ** (365 / manual_dte) - 1) * 100

        m1, m2, m3 = st.columns(3)

        m1.metric("Period Return", f"{period_return * 100:.2f}%")
        m2.metric("Simple Annualized", f"{simple_annualized:.2f}%")
        m3.metric("CAGR", f"{manual_cagr:.2f}%")


# =====================================================
# Manual Put Ladder Calculator
# =====================================================
st.divider()
st.subheader("🪜 Manual Put Ladder Calculator")

st.write(
    "Enter total capital, strikes and premiums. "
    "Choose allocation structure and the app will automatically show "
    "3 legs or 4 legs. DTE uses the loaded option-chain expiration."
)

# IMPORTANT:
# This selectbox must be OUTSIDE the form,
# so Streamlit redraws 3 or 4 legs immediately when you change it.
ladder_structure = st.selectbox(
    "Allocation Structure",
    options=["10/20/30/40", "15/30/55"],
    key="ladder_structure_select"
)

last_price = st.session_state.get("option_stock_price")

if last_price is None:
    st.info("Load an option chain first so the ladder can use the current stock price.")
    last_price = 100.0  # fallback default until option chain is loaded

if ladder_structure == "10/20/30/40":
    default_allocations = [10.0, 20.0, 30.0, 40.0]

    default_strikes = [
        round(last_price * (0.5 ** 1), 2),
        round(last_price * (0.7 ** 2), 2),
        round(last_price * (0.7 ** 3), 2),
        round(last_price * (0.7 ** 4), 2),
    ]

    default_premiums = [1.50, 0.42, 0.40, 0.30]

else:
    default_allocations = [15.0, 30.0, 55.0]

    default_strikes = [
        round(last_price * (0.5 ** 1), 2),
        round(last_price * (0.7 ** 2), 2),
        round(last_price * (0.7 ** 3), 2),
    ]

    default_premiums = [1.50, 0.42, 0.30]

with st.form("manual_put_ladder_form"):
    top_col1, top_col2 = st.columns(2)

    ladder_total_capital = top_col1.number_input(
        "Total Capital / Collateral",
        min_value=0.0,
        value=122000.0,
        step=1000.0,
        placeholder="Example: 100000"
    )

    loaded_option_dte = st.session_state.get("option_dte")

    if loaded_option_dte is not None:
        ladder_dte_default = int(loaded_option_dte)
        dte_help_text = "Using the same DTE as the loaded option-chain expiration."
    else:
        ladder_dte_default = 30
        dte_help_text = "Load an option chain first to use its DTE automatically."

    ladder_dte = top_col2.number_input(
        "DTE",
        min_value=1,
        value=ladder_dte_default,
        step=1,
        help=dte_help_text
    )

    st.markdown("### Enter Ladder Legs")

    header_cols = st.columns([1, 1, 1, 1])
    header_cols[0].markdown("**Leg**")
    header_cols[1].markdown("**Allocation %**")
    header_cols[2].markdown("**Strike**")
    header_cols[3].markdown("**Premium**")

    ladder_rows = []

    for i, allocation in enumerate(default_allocations):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

        c1.write(f"Leg {i + 1}")

        allocation_pct = c2.number_input(
            f"Allocation % {i + 1}",
            min_value=0.0,
            max_value=100.0,
            value=allocation,
            step=1.0,
            key=f"{ladder_structure}_allocation_pct_{i}"
        )

        strike = c3.number_input(
            f"Strike {i + 1}",
            min_value=0.0,
            value=default_strikes[i],
            step=0.5,
            key=f"{ladder_structure}_strike_{i}"
        )

        premium = c4.number_input(
            f"Premium {i + 1}",
            min_value=0.0,
            value=default_premiums[i],
            step=0.01,
            key=f"{ladder_structure}_premium_{i}"
        )

        ladder_rows.append({
            "Leg": i + 1,
            "Allocation %": allocation_pct,
            "Strike": strike,
            "Premium": premium
        })

    calculate_ladder = st.form_submit_button("Calculate Put Ladder")


if calculate_ladder:
    ladder_df = pd.DataFrame(ladder_rows)

    allocation_sum = ladder_df["Allocation %"].sum()

    if ladder_total_capital <= 0:
        st.error("Total Capital / Collateral must be greater than 0.")

    elif ladder_dte <= 0:
        st.error("DTE must be greater than 0.")

    elif abs(allocation_sum - 100) > 0.01:
        st.error(f"Allocation must sum to 100%. Current sum: {allocation_sum:.2f}%")

    elif (ladder_df["Strike"] <= 0).any():
        st.error("All strikes must be greater than 0.")

    else:
        base_ladder_df = build_put_ladder_df(
            ladder_rows=ladder_rows,
            total_capital=ladder_total_capital,
            dte=ladder_dte
        )

        st.session_state.put_ladder_base_df = base_ladder_df
        st.session_state.put_ladder_params = {
            "total_capital": ladder_total_capital,
            "dte": ladder_dte,
            "structure": ladder_structure
        }


# Display ladder results and allow extra contracts from unused capital
if st.session_state.put_ladder_base_df is not None and st.session_state.put_ladder_params is not None:
    params = st.session_state.put_ladder_params
    total_capital = params["total_capital"]
    dte = params["dte"]

    base_ladder_df = st.session_state.put_ladder_base_df.copy()
    base_summary = summarize_ladder(base_ladder_df, total_capital, dte)

    display_ladder_summary(base_summary, title="Base Ladder Summary")
    display_ladder_details(base_ladder_df, title="Base Ladder Details", download_key="base_ladder_details")

    st.divider()
    st.subheader("➕ Add Extra Contracts From Unused Capital")

    st.write(
        "Edit only the **Extra Contracts** column. "
        "The app will add these contracts to the base ladder and recalculate the full result."
    )

    editable_df = base_ladder_df[
        [
            "Leg",
            "Allocation %",
            "Strike",
            "Premium",
            "Contracts",
            "Actual Collateral",
            "Premium Cash",
            "Period Return %",
            "Net Assignment Price"
        ]
    ].copy()

    editable_df["Extra Contracts"] = 0

    edited_extra_df = st.data_editor(
        editable_df,
        use_container_width=True,
        height=250,
        disabled=[
            "Leg",
            "Allocation %",
            "Strike",
            "Premium",
            "Contracts",
            "Actual Collateral",
            "Premium Cash",
            "Period Return %",
            "Net Assignment Price"
        ],
        column_config={
            "Extra Contracts": st.column_config.NumberColumn(
                "Extra Contracts",
                min_value=0,
                step=1,
                help="Add contracts using unused capital."
            )
        },
        key="extra_contracts_editor"
    )

    final_ladder_df = base_ladder_df.copy()
    final_ladder_df["Extra Contracts"] = edited_extra_df["Extra Contracts"].fillna(0).astype(int)
    final_ladder_df["Final Contracts"] = (
        final_ladder_df["Contracts"] + final_ladder_df["Extra Contracts"]
    )

    final_ladder_df = recalc_ladder_with_contracts(
        ladder_df=final_ladder_df,
        total_capital=total_capital,
        dte=dte,
        contracts_col="Final Contracts"
    )

    final_summary = summarize_ladder(final_ladder_df, total_capital, dte)

    if final_summary["total_unused_capital"] < -0.01:
        st.error(
            "You added too many contracts. "
            f"Used collateral exceeds total capital by ${abs(final_summary['total_unused_capital']):,.2f}."
        )
    else:
        display_ladder_summary(final_summary, title="Final Ladder Summary After Extra Contracts")
        display_ladder_details(final_ladder_df, title="Final Ladder Details After Extra Contracts", download_key="final_ladder_details_after_extra_contracts")

    st.markdown(
        """
        ### Formulas

        **Contracts**

        `Contracts = floor(Target Capital / (Strike * 100))`

        **Final Contracts**

        `Final Contracts = Base Contracts + Extra Contracts`

        **Premium Cash**

        `Premium Cash = Final Contracts * Premium * 100`

        **Period Return per leg**

        `Period Return % = Premium Cash / Actual Collateral * 100`

        **Weighted Period Return**

        `Weighted Period Return = Total Premium Cash / Total Actual Collateral`

        **CAGR**

        `CAGR = ((1 + Period Return) ** (365 / DTE) - 1) * 100`

        **Net Assignment Price**

        `Net Assignment Price = Strike - Premium`

        **Average Assignment Price**

        `Avg Assignment Price = sum(Strike * Actual Weight %) / sum(Actual Weight %)`
        """
    )

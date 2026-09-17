# streamlit_app_manual_calculator.py

import streamlit as st
import pandas as pd
import numpy as np


# =====================================================
# Page setup
# =====================================================
st.set_page_config(
    page_title="Options Calculators",
    layout="wide"
)

st.title("📊 Options Calculators")
st.write(
    "Standalone manual calculators. "
    "No Yahoo Finance, yfinance, ticker lookup, or external market-data requests."
)


# =====================================================
# Session state
# =====================================================
if "put_ladder_base_df" not in st.session_state:
    st.session_state.put_ladder_base_df = None

if "put_ladder_params" not in st.session_state:
    st.session_state.put_ladder_params = None


# =====================================================
# Helper functions
# =====================================================
def calc_cagr_by_premium(premium, capital_base, dte):
    """
    CAGR = (1 + premium / capital_base) ** (365 / DTE) - 1
    Returned as percent.
    """
    if pd.isna(premium) or pd.isna(capital_base):
        return np.nan

    if premium < 0 or capital_base <= 0 or dte <= 0:
        return np.nan

    return ((1 + premium / capital_base) ** (365 / dte) - 1) * 100


def build_put_ladder_df(ladder_rows, total_capital, dte):
    """
    Build ladder dataframe from manual rows.

    Required columns:
    Leg, Allocation %, Strike, Premium
    """
    ladder_df = pd.DataFrame(ladder_rows).copy()

    ladder_df["Allocation Weight"] = ladder_df["Allocation %"] / 100
    ladder_df["Target Capital"] = (
        total_capital * ladder_df["Allocation Weight"]
    )

    ladder_df["Collateral Per Contract"] = (
        ladder_df["Strike"] * 100
    )

    ladder_df["Contracts"] = np.floor(
        ladder_df["Target Capital"] /
        ladder_df["Collateral Per Contract"]
    ).astype(int)

    return recalc_ladder_with_contracts(
        ladder_df=ladder_df,
        total_capital=total_capital,
        dte=dte,
        contracts_col="Contracts"
    )


def recalc_ladder_with_contracts(
    ladder_df,
    total_capital,
    dte,
    contracts_col="Contracts"
):
    """
    Recalculate ladder metrics using the selected contracts column.
    """
    df = ladder_df.copy()

    df[contracts_col] = (
        df[contracts_col]
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )

    df["Final Contracts"] = df[contracts_col]

    df["Actual Collateral"] = (
        df["Final Contracts"] *
        df["Strike"] *
        100
    )

    df["Actual Weight %"] = np.where(
        total_capital > 0,
        df["Actual Collateral"] / total_capital * 100,
        np.nan
    )

    df["Unused Capital By Target"] = (
        df["Target Capital"] -
        df["Actual Collateral"]
    )

    df["Premium Cash"] = (
        df["Final Contracts"] *
        df["Premium"] *
        100
    )

    df["Period Return %"] = np.where(
        df["Actual Collateral"] > 0,
        df["Premium Cash"] /
        df["Actual Collateral"] *
        100,
        np.nan
    )

    df["Simple Annualized %"] = (
        df["Period Return %"] *
        365 /
        dte
    )

    df["CAGR %"] = np.where(
        df["Period Return %"].notna(),
        (
            (1 + df["Period Return %"] / 100) **
            (365 / dte) -
            1
        ) * 100,
        np.nan
    )

    df["Net Assignment Price"] = (
        df["Strike"] -
        df["Premium"]
    )

    df["Shares If Assigned"] = (
        df["Final Contracts"] *
        100
    )

    return df


def summarize_ladder(ladder_df, total_capital, dte):
    """
    Return basket-level summary metrics.
    """
    total_actual_collateral = (
        ladder_df["Actual Collateral"].sum()
    )

    total_premium_cash = (
        ladder_df["Premium Cash"].sum()
    )

    total_unused_capital = (
        total_capital -
        total_actual_collateral
    )

    total_contracts = (
        ladder_df["Final Contracts"].sum()
    )

    total_shares_if_assigned = (
        ladder_df["Shares If Assigned"].sum()
    )

    if total_actual_collateral > 0:
        weighted_period_return = (
            total_premium_cash /
            total_actual_collateral
        )

        weighted_simple_annualized = (
            weighted_period_return *
            365 /
            dte
        )

        weighted_cagr = (
            (1 + weighted_period_return) **
            (365 / dte) -
            1
        )
    else:
        weighted_period_return = np.nan
        weighted_simple_annualized = np.nan
        weighted_cagr = np.nan

    valid_weights = (
        ladder_df["Actual Weight %"].notna() &
        (ladder_df["Actual Weight %"] > 0)
    )

    if valid_weights.any():
        weight_sum = (
            ladder_df
            .loc[valid_weights, "Actual Weight %"]
            .sum()
        )

        avg_assignment_price = (
            (
                ladder_df.loc[valid_weights, "Strike"] *
                ladder_df.loc[
                    valid_weights,
                    "Actual Weight %"
                ]
            ).sum()
            /
            weight_sum
        )
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
        "avg_assignment_price": avg_assignment_price,
    }


def safe_money(value):
    if pd.isna(value):
        return "-"
    return f"${value:,.2f}"


def safe_percent(value):
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def display_ladder_summary(
    summary,
    title="Ladder Summary"
):
    st.markdown(f"### {title}")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Total Contracts",
        f"{int(summary['total_contracts']):,}"
    )

    s2.metric(
        "Total Premium",
        safe_money(
            summary["total_premium_cash"]
        )
    )

    s3.metric(
        "Used Collateral",
        safe_money(
            summary["total_actual_collateral"]
        )
    )

    s4.metric(
        "Unused Capital",
        safe_money(
            summary["total_unused_capital"]
        )
    )

    s5, s6, s7, s8 = st.columns(4)

    s5.metric(
        "Weighted Period Return",
        safe_percent(
            summary["weighted_period_return"]
        )
    )

    s6.metric(
        "Weighted Simple Annualized",
        safe_percent(
            summary["weighted_simple_annualized"]
        )
    )

    s7.metric(
        "Weighted CAGR",
        safe_percent(
            summary["weighted_cagr"]
        )
    )

    s8.metric(
        "Avg Assignment Price",
        safe_money(
            summary["avg_assignment_price"]
        )
    )


def display_ladder_details(
    ladder_df,
    title="Ladder Details",
    download_key="ladder_csv"
):
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
        "Shares If Assigned",
    ]

    existing_cols = [
        col
        for col in display_cols
        if col in ladder_df.columns
    ]

    ladder_display = (
        ladder_df[existing_cols]
        .copy()
    )

    for col in ladder_display.columns:
        if pd.api.types.is_numeric_dtype(
            ladder_display[col]
        ):
            ladder_display[col] = (
                ladder_display[col]
                .round(2)
            )

    st.dataframe(
        ladder_display,
        use_container_width=True,
        height=350
    )

    ladder_csv = (
        ladder_display
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download Ladder CSV",
        data=ladder_csv,
        file_name=f"{download_key}.csv",
        mime="text/csv",
        key=f"download_{download_key}"
    )


# =====================================================
# Manual CAGR Calculator
# =====================================================
st.subheader("🧮 Manual CAGR Calculator")

st.write(
    "Enter Strike / Capital Base, Premium and DTE manually."
)

with st.form("manual_cagr_form"):
    calc_col1, calc_col2, calc_col3 = (
        st.columns(3)
    )

    manual_strike = (
        calc_col1.number_input(
            "Strike / Capital Base",
            min_value=0.0,
            value=None,
            step=0.01,
            placeholder="Example: 50"
        )
    )

    manual_premium = (
        calc_col2.number_input(
            "Premium",
            min_value=0.0,
            value=None,
            step=0.01,
            placeholder="Example: 2"
        )
    )

    manual_dte = (
        calc_col3.number_input(
            "DTE",
            min_value=1,
            value=None,
            step=1,
            placeholder="Example: 84"
        )
    )

    calculate_button = (
        st.form_submit_button(
            "Calculate CAGR"
        )
    )


if calculate_button:
    if (
        manual_strike is None or
        manual_premium is None or
        manual_dte is None
    ):
        st.error(
            "Please enter Strike, Premium and DTE."
        )

    elif manual_strike <= 0:
        st.error(
            "Strike / Capital Base must be greater than 0."
        )

    elif manual_dte <= 0:
        st.error(
            "DTE must be greater than 0."
        )

    else:
        period_return = (
            manual_premium /
            manual_strike
        )

        simple_annualized = (
            period_return *
            365 /
            manual_dte *
            100
        )

        manual_cagr = (
            (1 + period_return) **
            (365 / manual_dte) -
            1
        ) * 100

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Period Return",
            f"{period_return * 100:.2f}%"
        )

        m2.metric(
            "Simple Annualized",
            f"{simple_annualized:.2f}%"
        )

        m3.metric(
            "CAGR",
            f"{manual_cagr:.2f}%"
        )


# =====================================================
# Manual Put Ladder Calculator
# =====================================================
st.divider()
st.subheader("🪜 Manual Put Ladder Calculator")

st.write(
    "Enter everything manually: total capital, DTE, "
    "allocation structure, strike and premium for each leg."
)

ladder_structure = st.selectbox(
    "Allocation Structure",
    options=[
        "10/20/30/40",
        "15/30/55"
    ],
    key="ladder_structure_select"
)

if ladder_structure == "10/20/30/40":
    default_allocations = [
        10.0,
        20.0,
        30.0,
        40.0
    ]

    default_strikes = [
        50.0,
        40.0,
        30.0,
        20.0
    ]

    default_premiums = [
        1.50,
        0.90,
        0.50,
        0.30
    ]

else:
    default_allocations = [
        15.0,
        30.0,
        55.0
    ]

    default_strikes = [
        50.0,
        35.0,
        20.0
    ]

    default_premiums = [
        1.50,
        0.70,
        0.30
    ]


with st.form("manual_put_ladder_form"):
    top_col1, top_col2 = (
        st.columns(2)
    )

    ladder_total_capital = (
        top_col1.number_input(
            "Total Capital / Collateral",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            placeholder="Example: 100000"
        )
    )

    ladder_dte = (
        top_col2.number_input(
            "DTE",
            min_value=1,
            value=365,
            step=1,
            help="Enter the option DTE manually."
        )
    )

    st.markdown(
        "### Enter Ladder Legs"
    )

    header_cols = (
        st.columns([1, 1, 1, 1])
    )

    header_cols[0].markdown("**Leg**")
    header_cols[1].markdown("**Allocation %**")
    header_cols[2].markdown("**Strike**")
    header_cols[3].markdown("**Premium**")

    ladder_rows = []

    for i, allocation in enumerate(
        default_allocations
    ):
        c1, c2, c3, c4 = (
            st.columns([1, 1, 1, 1])
        )

        c1.write(f"Leg {i + 1}")

        allocation_pct = (
            c2.number_input(
                f"Allocation % {i + 1}",
                min_value=0.0,
                max_value=100.0,
                value=allocation,
                step=1.0,
                key=(
                    f"{ladder_structure}_"
                    f"allocation_pct_{i}"
                )
            )
        )

        strike = (
            c3.number_input(
                f"Strike {i + 1}",
                min_value=0.01,
                value=default_strikes[i],
                step=0.5,
                key=(
                    f"{ladder_structure}_"
                    f"strike_{i}"
                )
            )
        )

        premium = (
            c4.number_input(
                f"Premium {i + 1}",
                min_value=0.0,
                value=default_premiums[i],
                step=0.01,
                key=(
                    f"{ladder_structure}_"
                    f"premium_{i}"
                )
            )
        )

        ladder_rows.append({
            "Leg": i + 1,
            "Allocation %": allocation_pct,
            "Strike": strike,
            "Premium": premium,
        })

    calculate_ladder = (
        st.form_submit_button(
            "Calculate Put Ladder"
        )
    )


if calculate_ladder:
    ladder_df = pd.DataFrame(
        ladder_rows
    )

    allocation_sum = (
        ladder_df["Allocation %"].sum()
    )

    if ladder_total_capital <= 0:
        st.error(
            "Total Capital / Collateral must be greater than 0."
        )

    elif ladder_dte <= 0:
        st.error(
            "DTE must be greater than 0."
        )

    elif abs(
        allocation_sum - 100
    ) > 0.01:
        st.error(
            "Allocation must sum to 100%. "
            f"Current sum: {allocation_sum:.2f}%"
        )

    elif (
        ladder_df["Strike"] <= 0
    ).any():
        st.error(
            "All strikes must be greater than 0."
        )

    else:
        base_ladder_df = (
            build_put_ladder_df(
                ladder_rows=ladder_rows,
                total_capital=ladder_total_capital,
                dte=ladder_dte
            )
        )

        st.session_state.put_ladder_base_df = (
            base_ladder_df
        )

        st.session_state.put_ladder_params = {
            "total_capital":
                ladder_total_capital,
            "dte":
                ladder_dte,
            "structure":
                ladder_structure,
        }


# =====================================================
# Ladder Results
# =====================================================
if (
    st.session_state.put_ladder_base_df
    is not None
    and
    st.session_state.put_ladder_params
    is not None
):
    params = (
        st.session_state
        .put_ladder_params
    )

    total_capital = (
        params["total_capital"]
    )

    dte = params["dte"]

    base_ladder_df = (
        st.session_state
        .put_ladder_base_df
        .copy()
    )

    base_summary = summarize_ladder(
        base_ladder_df,
        total_capital,
        dte
    )

    display_ladder_summary(
        base_summary,
        title="Base Ladder Summary"
    )

    display_ladder_details(
        base_ladder_df,
        title="Base Ladder Details",
        download_key="base_ladder_details"
    )

    st.divider()
    st.subheader(
        "➕ Add Extra Contracts From Unused Capital"
    )

    st.write(
        "Edit only the Extra Contracts column. "
        "The app will add them to the base ladder "
        "and recalculate the full result."
    )

    editable_df = (
        base_ladder_df[
            [
                "Leg",
                "Allocation %",
                "Strike",
                "Premium",
                "Contracts",
                "Actual Collateral",
                "Premium Cash",
                "Period Return %",
                "Net Assignment Price",
            ]
        ]
        .copy()
    )

    editable_df[
        "Extra Contracts"
    ] = 0

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
            "Net Assignment Price",
        ],
        column_config={
            "Extra Contracts":
                st.column_config.NumberColumn(
                    "Extra Contracts",
                    min_value=0,
                    step=1,
                    help=(
                        "Add contracts using "
                        "unused capital."
                    )
                )
        },
        key="extra_contracts_editor"
    )

    final_ladder_df = (
        base_ladder_df.copy()
    )

    final_ladder_df[
        "Extra Contracts"
    ] = (
        edited_extra_df[
            "Extra Contracts"
        ]
        .fillna(0)
        .astype(int)
    )

    final_ladder_df[
        "Final Contracts"
    ] = (
        final_ladder_df["Contracts"] +
        final_ladder_df["Extra Contracts"]
    )

    final_ladder_df = (
        recalc_ladder_with_contracts(
            ladder_df=final_ladder_df,
            total_capital=total_capital,
            dte=dte,
            contracts_col="Final Contracts"
        )
    )

    final_summary = summarize_ladder(
        final_ladder_df,
        total_capital,
        dte
    )

    if (
        final_summary[
            "total_unused_capital"
        ] < -0.01
    ):
        st.error(
            "You added too many contracts. "
            "Used collateral exceeds total "
            f"capital by "
            f"${abs(final_summary['total_unused_capital']):,.2f}."
        )

    else:
        display_ladder_summary(
            final_summary,
            title=(
                "Final Ladder Summary "
                "After Extra Contracts"
            )
        )

        display_ladder_details(
            final_ladder_df,
            title=(
                "Final Ladder Details "
                "After Extra Contracts"
            ),
            download_key=(
                "final_ladder_details_"
                "after_extra_contracts"
            )
        )

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

**Simple Annualized Return**

`Simple Annualized % = Period Return % * 365 / DTE`

**CAGR**

`CAGR = ((1 + Period Return) ** (365 / DTE) - 1) * 100`

**Net Assignment Price**

`Net Assignment Price = Strike - Premium`

**Average Assignment Price**

`Avg Assignment Price = sum(Strike * Actual Weight %) / sum(Actual Weight %)`
        """
    )

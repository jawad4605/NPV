import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.optimize import minimize

# ---------------------------------------------------------------------------------
# PAGE SETUP & STYLES
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="Free LCOH Model", layout="wide")

# Optional CSS to give an Excel-like feel
st.markdown(
    """
    <style>
    .param-header {
        font-weight: 600;
        background-color: #f0f2f6;
        padding: 6px;
        border-bottom: 2px solid #ccc;
        text-align: center;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 6px;
        border-bottom: 2px solid #ccc;
        padding-bottom: 3px;
    }
    .calc-cell {
        background-color: #fff3cd;
        border-radius: 4px;
        padding: 6px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Free LCOH Model - Advanced & Complete")

# ---------------------------------------------------------------------------------
# PARAMETER LIST (matching your screenshot exactly)
# Columns: Parameter | Green H2 | min | max | In the optimization
# ---------------------------------------------------------------------------------
parameters = [
    {
        "key": "capex_mw_yr",
        "label": "CAPEX [$/Mw/yr]",
        "default_value": 3242.0,
        "min_value": 0.0,
        "max_value": 999999999.0,
        "in_optimization": True
    },
    {
        "key": "opex_mw_yr",
        "label": "OPEX [$/Mw/yr]",
        "default_value": 90000.0,
        "min_value": 0.0,
        "max_value": 999999999.0,
        "in_optimization": True
    },
    {
        "key": "annual_h2_prod",
        "label": "Annual H2 Production [Kg/yr]",
        "default_value": 876000.0,
        "min_value": 0.0,
        "max_value": 999999999.0,
        "in_optimization": False
    },
    {
        "key": "plant_size_mw",
        "label": "Plant Size [Mw]",
        "default_value": 100.0,
        "min_value": 0.0,
        "max_value": 999999.0,
        "in_optimization": False
    },
    {
        "key": "plant_life",
        "label": "Plant Lifetime [yr]",
        "default_value": 20.0,
        "min_value": 1.0,
        "max_value": 30.0,
        "in_optimization": False
    },
    {
        "key": "discount_rate",
        "label": "Discount rate [%]",
        "default_value": 5.0,
        "min_value": 0.0,
        "max_value": 15.0,
        "in_optimization": False
    },
    {
        "key": "capacity_factor",
        "label": "Capacity Factor [%]",
        "default_value": 90.0,
        "min_value": 0.0,
        "max_value": 100.0,
        "in_optimization": True
    },
    {
        "key": "h2_efficiency_1",
        "label": "H2 Efficiency [kWh/kg] (1)",
        "default_value": 50.0,
        "min_value": 0.0,
        "max_value": 9999.0,
        "in_optimization": False
    },
    {
        "key": "h2_efficiency_2",
        "label": "H2 Efficiency [kWh/kg] (2)",
        "default_value": 70.0,
        "min_value": 0.0,
        "max_value": 9999.0,
        "in_optimization": False
    },
    {
        "key": "electricity_cost",
        "label": "Electricity Cost [$/Mwh]",
        "default_value": 3.0,
        "min_value": 1.0,
        "max_value": 700.0,
        "in_optimization": True
    },
    {
        "key": "crf",
        "label": "CRF",
        "default_value": 0.094392926,
        "min_value": 0.0,
        "max_value": 1.0,
        "in_optimization": False
    },
    {
        "key": "dcf_factor",
        "label": "DCF factor",
        "default_value": 10.59401425,
        "min_value": 0.0,
        "max_value": 999999.0,
        "in_optimization": False
    },
    {
        "key": "h2_selling_price",
        "label": "H2 Selling Price [$/kg]",
        "default_value": 3.5,
        "min_value": 0.0,
        "max_value": 999999.0,
        "in_optimization": True
    },
    {
        "key": "carbon_tax",
        "label": "Carbon Tax [$/ton co2]",
        "default_value": 0.0,
        "min_value": 0.0,
        "max_value": 999999.0,
        "in_optimization": True
    },
    {
        "key": "tax_credit",
        "label": "Tax Credit [$/kgH2]",
        "default_value": 0.0,
        "min_value": 0.0,
        "max_value": 999999.0,
        "in_optimization": True
    },
    {
        "key": "h2_storage_cost",
        "label": "Hydrogen Storage Cost [$/kg]",
        "default_value": 1.0,
        "min_value": 1.0,
        "max_value": 700.0,
        "in_optimization": False
    },
    {
        "key": "h2_transport_cost",
        "label": "Hydrogen Transportation Cost [$/kg]",
        "default_value": 0.0,
        "min_value": 0.0,
        "max_value": 999999.0,
        "in_optimization": False
    },
]

# Dictionaries to store the user’s numeric inputs and optimization info
user_values = {}
opt_flags = {}

# ---------------------------------------------------------------------------------
# TABBED LAYOUT
# We'll create two tabs:
#   1) "Model & Optimization" for parameters, results, and optimization
#   2) "Plots & Sensitivity" for graphs of how NPV changes with each decision variable
# ---------------------------------------------------------------------------------
tabs = st.tabs(["Model & Optimization", "Plots & Sensitivity"])

# -----------------------------------------------
# TAB 1: MODEL & OPTIMIZATION
# -----------------------------------------------
with tabs[0]:
    left_col, right_col = st.columns([2, 1], gap="large")

    # -----------------------------------
    # LEFT COLUMN: Parameter Table
    # -----------------------------------
    with left_col:
        st.subheader("Parameters")

        # Build table header
        hdr = st.columns([2.5, 1.2, 1, 1, 1.2])
        with hdr[0]:
            st.markdown("<div class='param-header'>Parameter</div>", unsafe_allow_html=True)
        with hdr[1]:
            st.markdown("<div class='param-header'>Green H2</div>", unsafe_allow_html=True)
        with hdr[2]:
            st.markdown("<div class='param-header'>min</div>", unsafe_allow_html=True)
        with hdr[3]:
            st.markdown("<div class='param-header'>max</div>", unsafe_allow_html=True)
        with hdr[4]:
            st.markdown("<div class='param-header'>In the optimization</div>", unsafe_allow_html=True)

        # Create a row for each parameter
        for param in parameters:
            row_cols = st.columns([2.5, 1.2, 1, 1, 1.2])
            with row_cols[0]:
                st.write(param["label"])
            with row_cols[1]:
                val = st.number_input(
                    label=" ",
                    value=float(param["default_value"]),
                    min_value=float(param["min_value"]),
                    max_value=float(param["max_value"]),
                    key=f"{param['key']}_val"
                )
            with row_cols[2]:
                mn = st.number_input(
                    label=" ",
                    value=float(param["min_value"]),
                    key=f"{param['key']}_min"
                )
            with row_cols[3]:
                mx = st.number_input(
                    label=" ",
                    value=float(param["max_value"]),
                    key=f"{param['key']}_max"
                )
            with row_cols[4]:
                in_opt = st.checkbox(
                    label=" ",
                    value=param["in_optimization"],
                    key=f"{param['key']}_opt"
                )
            
            user_values[param["key"]] = val
            opt_flags[param["key"]] = {
                "min": mn,
                "max": mx,
                "in_opt": in_opt
            }

    # -----------------------------------
    # RIGHT COLUMN: Components & Results
    # -----------------------------------
    with right_col:
        st.subheader("Components & Results")

        def calculate_model(vals):
            """
            Placeholder formulas for LCOH, NPV, etc.
            Replace with your exact Excel logic.
            """
            capex_yr = vals["capex_mw_yr"]
            opex_yr = vals["opex_mw_yr"]
            annual_prod = vals["annual_h2_prod"]
            plant_life = vals["plant_life"]
            discount_rate = vals["discount_rate"] / 100.0
            cap_factor = vals["capacity_factor"] / 100.0
            # Two separate H2 efficiencies from the screenshot
            eff1 = vals["h2_efficiency_1"]
            eff2 = vals["h2_efficiency_2"]
            elec_cost = vals["electricity_cost"]
            crf = vals["crf"]
            dcf_factor = vals["dcf_factor"]
            h2_price = vals["h2_selling_price"]
            carbon_tax_ton = vals["carbon_tax"]
            tax_credit = vals["tax_credit"]
            storage_cost = vals["h2_storage_cost"]
            transport_cost = vals["h2_transport_cost"]

            # Avoid division by zero
            if annual_prod <= 0:
                annual_prod = 1e-9

            # ------------------------
            # COMPONENT [$/KgH2]
            # Example logic (placeholder)
            capex_per_kg = (capex_yr * crf) / annual_prod
            opex_per_kg = opex_yr / annual_prod
            # Suppose total efficiency is sum of both? Or average? You decide:
            elec_per_kg = (eff1 + eff2) * (elec_cost / 1000.0)
            carbon_tax_per_kg = carbon_tax_ton / 1000.0  # if $/ton => $/kg
            # LCOH = sum minus tax credit
            lcoh = capex_per_kg + opex_per_kg + elec_per_kg + carbon_tax_per_kg + storage_cost + transport_cost - tax_credit
            money_check = "Money" if lcoh < h2_price else "No Money"

            # ------------------------
            # COMPONENT [$]
            revenue = h2_price * annual_prod
            total_cost = lcoh * annual_prod
            profit = revenue - total_cost
            # Simple placeholder for NPV
            npv = profit * dcf_factor  # e.g. single-lump approach
            # Basic placeholders for payback & ROI
            payback = 0.0
            roi = 0.0
            if total_cost != 0:
                roi = (profit / total_cost) * 100.0
            if profit > 0:
                payback = 5.0  # dummy example

            return {
                # [$/KgH2]
                "capex_per_kg": capex_per_kg,
                "opex_per_kg": opex_per_kg,
                "elec_per_kg": elec_per_kg,
                "carbon_tax_per_kg": carbon_tax_per_kg,
                "lcoh": lcoh,
                "money_check": money_check,
                # [$]
                "revenue": revenue,
                "cost": total_cost,
                "profit": profit,
                "npv": npv,
                "payback": payback,
                "roi": roi
            }

        # Calculate with current user inputs
        results = calculate_model(user_values)

        st.markdown("<div class='section-title'>Component [$/KgH2]</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="calc-cell">
            <strong>CAPEX:</strong> {results['capex_per_kg']:.6f}<br/>
            <strong>OPEX:</strong> {results['opex_per_kg']:.6f}<br/>
            <strong>Elec. Cost:</strong> {results['elec_per_kg']:.6f}<br/>
            <strong>Carbon Tax:</strong> {results['carbon_tax_per_kg']:.6f}<br/>
            <strong>LCOH:</strong> {results['lcoh']:.6f}<br/>
            Money/No Money: <strong>{results['money_check']}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='section-title'>Component [$]</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="calc-cell">
            <strong>Revenue:</strong> {results['revenue']:.2f}<br/>
            <strong>Cost:</strong> {results['cost']:.2f}<br/>
            <strong>Profit:</strong> {results['profit']:.2f}<br/>
            <strong>NPV:</strong> {results['npv']:.2f}<br/>
            <strong>Payback [yr]:</strong> {results['payback']:.2f}<br/>
            <strong>ROI [%]:</strong> {results['roi']:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------
    # OPTIMIZATION (Maximize NPV)
    # -----------------------------------
    st.markdown("---")
    st.subheader("Optimize NPV")

    # Minimizing negative NPV => maximizing NPV
    def objective(x, fixed_params, dv_keys):
        new_vals = fixed_params.copy()
        for i, k in enumerate(dv_keys):
            new_vals[k] = x[i]
        out = calculate_model(new_vals)
        return -out["npv"]

    # Optional constraint: LCOH < H2 Selling Price
    # If you have more constraints, define them similarly
    enforce_money = st.checkbox("Enforce LCOH < H2 Selling Price?", value=False)

    def money_constraint(x, fixed_params, dv_keys):
        new_vals = fixed_params.copy()
        for i, k in enumerate(dv_keys):
            new_vals[k] = x[i]
        out = calculate_model(new_vals)
        # LCOH < h2_selling_price => (h2_selling_price - LCOH) >= 0
        return new_vals["h2_selling_price"] - out["lcoh"]

    # Build the list of decision variables
    decision_keys = [p["key"] for p in parameters if opt_flags[p["key"]]["in_opt"]]
    x0 = []
    bounds = []
    for k in decision_keys:
        x0.append(user_values[k])
        bounds.append((opt_flags[k]["min"], opt_flags[k]["max"]))

    if st.button("Optimize NPV"):
        cons = []
        if enforce_money:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: money_constraint(x, user_values, decision_keys)
            })

        res = minimize(
            fun=objective,
            x0=np.array(x0),
            args=(user_values, decision_keys),
            method="SLSQP",
            bounds=bounds,
            constraints=cons
        )

        if res.success:
            st.success("Optimization successful!")
            final_params = user_values.copy()
            for i, k in enumerate(decision_keys):
                final_params[k] = res.x[i]
            final_res = calculate_model(final_params)

            st.write("**Optimized Decision Variables:**")
            for i, k in enumerate(decision_keys):
                st.write(f"- {k} = {res.x[i]:.4f}")

            st.markdown(
                f"""
                **Final NPV:** {final_res['npv']:.2f}  
                **Final LCOH:** {final_res['lcoh']:.6f}  
                **Money/No Money:** {final_res['money_check']}
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Optimization failed. Try adjusting bounds or constraints.")

# -----------------------------------------------
# TAB 2: PLOTS & SENSITIVITY
# -----------------------------------------------
with tabs[1]:
    st.header("Plots & Sensitivity Analysis")

    st.markdown("""
    Below, we vary each **decision variable** from its min to max in a series of steps,
    holding the other parameters fixed at their current values, and plot **NPV** to see
    how sensitive the model is to each variable.
    """)

    # We'll only do sensitivity for the "decision variables" that are in optimization
    if len(decision_keys) == 0:
        st.info("No variables are marked 'In the optimization.' Please select at least one in Tab 1.")
    else:
        # For each decision variable, we create a line chart: x=variable, y=NPV
        for k in decision_keys:
            # We'll sample from param's min to max in ~20 steps
            var_min = opt_flags[k]["min"]
            var_max = opt_flags[k]["max"]
            steps = np.linspace(var_min, var_max, 20)

            data_rows = []
            for val in steps:
                # Make a copy of user_values, override the single variable
                temp_params = user_values.copy()
                temp_params[k] = val
                out = calculate_model(temp_params)
                data_rows.append({"ParamValue": val, "NPV": out["npv"]})

            df_plot = pd.DataFrame(data_rows)
            chart = (
                alt.Chart(df_plot)
                .mark_line(point=True)
                .encode(
                    x=alt.X("ParamValue", title=f"{k}"),
                    y=alt.Y("NPV", title="NPV"),
                    tooltip=["ParamValue", "NPV"]
                )
                .properties(width=600, height=300, title=f"Sensitivity of NPV vs {k}")
            )
            st.altair_chart(chart, use_container_width=True)

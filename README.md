# Free LCOH Model

## Overview
This Streamlit app calculates and optimizes the **Levelized Cost of Hydrogen (LCOH)** and **Net Present Value (NPV)** for a hydrogen production plant. It takes user-defined parameters related to costs, efficiency, and production and performs cost calculations, profitability analysis, and optimization to maximize NPV.

---

## How the System Works

### 1. **User Input Parameters**
Users define key inputs such as:
- **CAPEX** (Capital Expenditure)
- **OPEX** (Operational Expenditure)
- **Plant Size**
- **Efficiency**
- **Electricity Cost**
- **Hydrogen Selling Price**
- **Carbon Tax** (if applicable)

Some parameters can be optimized, and users can set min/max bounds for them.

### 2. **LCOH & NPV Calculation**
#### **LCOH Calculation**
LCOH is calculated as the sum of all costs per kg of hydrogen, including:
- **CAPEX** and its amortization
- **OPEX** (fixed & variable costs)
- **Electricity Cost** (key driver of cost)
- **Carbon Tax & Tax Credits** (if applicable)

#### **NPV Calculation**
Net Present Value (NPV) is calculated as:
\[ NPV = \sum \frac{Revenue - Total Costs}{(1 + Discount Rate)^t} \]
Where revenue is derived from:
\[ Revenue = Hydrogen Selling Price \times Production \]

### 3. **Optimization (Maximizing NPV)**
The app uses **SciPy's `minimize()` function** to find the best values for selected variables while ensuring constraints, such as:
- Keeping **LCOH < Selling Price**
- Ensuring efficiency stays within defined bounds
- Minimizing CAPEX and OPEX where possible

### 4. **Sensitivity Analysis (Tab 2: Plots)**
- Generates **graphs** showing how NPV changes when varying one input while keeping others fixed.
- Helps users understand the impact of different variables on project profitability.

---

## **How Everything Links**
1. **User Inputs** → Define key cost and efficiency factors.
2. **Calculations** → Compute LCOH, NPV, and profitability.
3. **Optimization** → Adjust decision variables to find the most cost-effective solution.
4. **Plots** → Display sensitivity of NPV to key parameters.

The system adjusts decision variables to minimize costs and maximize profitability while ensuring constraints (e.g., cost remains below selling price). This helps decision-makers find optimal conditions for hydrogen production at the lowest cost while maintaining profitability.

---

## **Usage Instructions**
1. Run the **Streamlit** app.
2. Input **your project parameters**.
3. Click **"Optimize"** to get the best values.
4. View **sensitivity analysis** to understand cost drivers.
5. Make **data-driven decisions** for hydrogen production.

---

## **License**
This model is free to use under the [MIT License](LICENSE).

---

**Developed by: jawad ahmad**


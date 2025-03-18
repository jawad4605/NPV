How the Free LCOH Model Works
This Streamlit app calculates and optimizes the Levelized Cost of Hydrogen (LCOH) and Net Present Value (NPV) for a hydrogen production plant. It takes user-defined parameters related to costs, efficiency, and production and performs cost calculations, profitability analysis, and optimization to maximize NPV.

How Inputs & System Work Together
User Inputs Parameters

Users define key inputs like CAPEX, OPEX, plant size, efficiency, electricity cost, etc.

Some parameters can be optimized, and users set min/max bounds for them.

LCOH & NPV Calculation

LCOH: Sum of costs per kg of H₂ (CAPEX, OPEX, electricity, carbon tax, etc.) minus tax credits.

NPV: Revenue (H₂ Selling Price × Production) minus total costs, discounted over plant life.

Optimization (Maximizing NPV)

Uses SciPy's minimize() function to find the best values for selected variables while keeping within constraints (e.g., ensuring LCOH < Selling Price).

Sensitivity Analysis (Tab 2: Plots)

Generates graphs showing how NPV changes when varying one input while keeping others fixed.

How Everything Links
User Inputs → Calculations (LCOH, NPV, Profitability) → Optimization (Finds Best Parameters) → Plots (Shows Sensitivity of NPV to Each Parameter).

The system adjusts decision variables to minimize costs and maximize profitability while ensuring constraints (e.g., cost below selling price).

This helps decision-makers find optimal conditions for hydrogen production at the lowest cost while maintaining profitability. 
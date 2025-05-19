# Weekly Regime
Overview:
The Weekly Regime project leverages L1 trend filtering to identify and classify economic regimes from market data, capturing shifts in market conditions that impact portfolio risk. Inspired by the paper “Identifying Economic Regimes: Reducing Downside Risks for University Endowments and Foundations,” this tool segments financial time series into distinct regimes, helping investors and portfolio managers adapt to changing market environments.

This project also includes an automated workflow that generates weekly regime signals and distributes them via email, ensuring stakeholders receive timely insights without manual intervention.

Key Features:

- Applies L1 trend filtering for structural break detection and smoothing in weekly financial return series.
- Classifies regimes into distinct states (e.g., expansion, contraction) based on trend signals and regime thresholds.
- Supports flexible configuration of parameters, including trend filtering lambda and regime classification thresholds.
- Built using Python with libraries such as NumPy, Pandas, CVXPY (for convex optimization), Matplotlib, and SMTP for email automation

Use Case:
This project is tailored for portfolio managers, risk analysts, and financial researchers seeking to integrate regime-based analysis into risk management frameworks and decision-making processes. The automated email delivery ensures consistent communication of market conditions, enabling more proactive portfolio adjustments.

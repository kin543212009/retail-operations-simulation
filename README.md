# retail-operations-simulation
Python simulation of queue dynamics, customer arrivals, service time, churn, and retail revenue.
# Retail Operations Simulation

A Python-based retail operations simulation project that models queue dynamics, customer arrivals, waiting time, customer churn, and daily revenue under changing demand conditions.

## Project Objective
This project evaluates how cashier configuration, queue rules, and fluctuating customer flow affect service efficiency and revenue performance in a convenience-store scenario.

## Business Problem
Retail stores need to balance:
- labour efficiency
- customer waiting time
- service quality
- revenue performance

This simulation was designed to assess whether current cashier capacity is sufficient during peak and off-peak periods.

## Scenario Setup
- 24-hour operation
- Off-peak arrival rate: 1 customer per minute
- Peak arrival rate: up to 3x off-peak
- Maximum 2 cash registers
- Open a new register when queue length reaches 4
- Close idle registers after 1 minute
- Customers leave when queue length exceeds 8

## Methodology
The model combines:
- Non-Homogeneous Poisson Process (NHPP)
- Thinning algorithm
- Exponential distribution for service time
- Lognormal distribution for customer spending
- Discrete-event simulation
- Variance reduction techniques:
  - Antithetic variates
  - Control variates
- 500 simulation runs with 95% confidence intervals

## Tech Stack
- Python
- NumPy
- SciPy
- Matplotlib

## Key Results
- Average waiting time: 0.4255 minutes
- Customer churn rate: 0.0073%
- Average daily total revenue: 97,192.94
- Average daily customers: 2,430

## Files
- `Q2.py`: main simulation script
- `requirements.txt`: required Python libraries

## How to Run
```bash
pip install -r requirements.txt
python Q2.py

# Project Value
This project demonstrates how simulation and statistical modelling can support data-driven decision-making in retail operations by balancing staffing efficiency, customer experience, and revenue outcomes.

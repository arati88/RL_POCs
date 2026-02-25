# POC-05: Revenue-Based Product Recommendation using Epsilon-Greedy

## Objective

This Proof of Concept (POC-05) demonstrates the **Exploration vs Exploitation tradeoff**
using an **Epsilon-Greedy Multi-Armed Bandit algorithm**.

Unlike traditional click-based bandits, this implementation optimizes **Revenue (₹)** instead of clicks.

The agent learns which product to recommend in order to **maximize expected revenue per recommendation**.

---

## Key Concept

In recommendation systems:

- High CTR ≠ High Revenue
- A lower CTR but higher-priced product can generate more revenue

This POC shows how an RL agent learns this tradeoff using exploration.

---

## Problem Setup

Simulated 5 products:

| Product       | Price (₹) | True CTR | Expected Revenue (CTR × Price) |
|--------------|-----------|----------|--------------------------------|
| Electronics  | 8,000     | 45%      | ₹3,600 |
| Clothing     | 2,500     | 30%      | ₹750 |
| Books        | 500       | 20%      | ₹100 |
| Sports       | 5,000     | 35%      | ₹1,750 |
| Home Decor   | 9,000     | 25%      | ₹2,250 |

The agent does NOT know CTR values.

It must learn expected revenue through interaction.

---

## Architecture

### Environment
- Simulates customer click behavior
- Returns revenue (₹price if clicked, else 0)

### Agent
- Uses **Epsilon-Greedy**
- Maintains:
  - `Q[product]` → estimated revenue
  - `N[product]` → number of times recommended
- Update rule (Incremental Mean):
    Q[a] ← Q[a] + (reward − Q[a]) / N[a]
Over time:
    Q[a] → CTR × Price

## Epsilon Strategies Implemented

1. **Fixed ε = 0.10**
2. **Linear Decay (1.0 → 0.01)**
3. **Exponential Decay (rate = 0.995)**

This allows comparison of exploration strategies.


## Metrics Tracked

- Cumulative Revenue
- Cumulative Clicks
- Cumulative Regret
- Exploration vs Exploitation mode
- Epsilon value over time

### Regret Definition
Regret = Optimal Expected Revenue − Actual Revenue


Cumulative regret measures performance gap vs optimal strategy.

---

## BDD Testing (Behave + Gherkin)

This project includes **Behavior-Driven Development tests** validating:

- Exploration actually occurs
- Q-values update correctly
- Agent converges to optimal product
- Regret growth rate decreases over time
- Exponential decay outperforms fixed epsilon

### Run Tests

python -m behave

Expected Output:
1 feature passed
4 scenarios passed
21 steps passed

## Output of Expllration vs Exploitation Tradeoff will have
- Product setup
- Strategy comparison
- Final revenue
- Regret values
- Best learned product

## Key Learnings

- Exploration is necessary to discover optimal actions
- Too much exploration reduces revenue
- Too little exploration risks suboptimal convergence
- Regret should grow sub-linearly over time
- Revenue optimization can change which product is optimal

## POC-04: Retail Coupon Optimization via Policy Gradient (REINFORCE)

### Overview

This POC demonstrates how a **Monte Carlo Policy Gradient (REINFORCE)** agent can learn a profitable coupon-issuance policy in a retail environment.

The agent interacts with a simulated customer base, choosing at each step whether to issue a coupon. It observes the resulting purchase behaviour and customer-state transition, receiving a profit-based reward signal.

The key business challenge it addresses: **short-term conversion vs. long-term coupon addiction**.  Giving coupons boosts immediate purchases but risks training customers to *only* buy when a coupon is present—eroding long-term profitability.

---

### Environment

#### Customer States

| Constant        | Value | Description                                              |
|-----------------|-------|----------------------------------------------------------|
| `NEUTRAL`       | 0     | Moderate interest; moderate coupon response              |
| `COUPON_ADDICT` | 1     | Rarely purchases without a coupon; high coupon response  |
| `NORMAL_BUYER`  | 2     | Buys regularly; small additional uplift from coupons     |

#### Actions

| Constant      | Value | Description            |
|---------------|-------|------------------------|
| `NO_COUPON`   | 0     | Do not issue a coupon  |
| `GIVE_COUPON` | 1     | Issue a coupon         |

#### Reward Function

```
If purchase occurs:
    reward = margin - coupon_cost   (if GIVE_COUPON)
    reward = margin                 (if NO_COUPON)
If no purchase:
    reward = 0
```

Defaults: `margin = 10.0`, `coupon_cost = 3.0`

#### Purchase Probabilities (defaults)

| State         | NoCoupon | GiveCoupon |
|---------------|----------|------------|
| Neutral       | 0.20     | 0.45       |
| CouponAddict  | 0.05     | 0.60       |
| NormalBuyer   | 0.65     | 0.70       |

#### Transition Dynamics (defaults)

| State → Action      | → Neutral | → CouponAddict | → NormalBuyer |
|---------------------|-----------|----------------|---------------|
| Neutral + NoCoupon  | 0.70      | 0.05           | 0.25          |
| Neutral + Coupon    | 0.35      | **0.45**       | 0.20          |
| Addict + NoCoupon   | **0.30**  | 0.60           | 0.10          |
| Addict + Coupon     | 0.05      | **0.85**       | 0.10          |
| Normal + NoCoupon   | 0.10      | 0.05           | **0.85**      |
| Normal + Coupon     | 0.15      | 0.15           | **0.70**      |

Bold entries highlight the dominant dynamics described in the rationale section below.

---

### Algorithm

**REINFORCE** (on-policy Monte Carlo policy gradient):

1. Run a full episode using the current stochastic (softmax) policy.
2. Compute discounted returns G_t for each time-step.
3. Normalise returns to reduce variance.
4. Update the policy parameters θ:

```
θ ← θ + α · G_t · (one_hot(a_t) − π(a_t|s_t))
```

Hyperparameter defaults: `α = 0.01`, `γ = 0.95`, `episodes = 3000`, `steps_per_episode = 50`.

---

### How to Run

```bash
# From the POC04 directory
cd POC04

# Run the training script
python Policy_gradient.py

# Run BDD tests
python -m behave
```

Expected console output (approximate):
```
Training RL Agent...

Episode 0, Profit: 158.0
Episode 500, Profit: 223.0
...

Training Complete.

===== STRATEGY COMPARISON =====

Strategy                  Average Profit per Step
RL Optimized Policy       5.116
Always Give Coupon        4.283
Never Give Coupon         4.367

Best Performing Strategy: RL Optimized Policy

===== LEARNED DECISION POLICY =====

Neutral         → GiveCoupon
CouponAddict    → GiveCoupon
NormalBuyer     → NoCoupon
```

---

### How to Tune

All parameters are module-level constants in `Policy_gradient.py` and can be overridden when constructing `RetailEnv`:

```python
env = RetailEnv(
    margin=15.0,           # Higher-margin product category
    coupon_cost=4.0,       # More expensive coupon promotion
    purchase_probs={...},  # Custom P(buy|state,action) dict
    transition_probs={...},# Custom T(next|state,action) dict
    rng_seed=0,            # Fix seed for reproducibility
)
```

**When to adjust probabilities:**
- Increase `P(buy|NEUTRAL, GIVE_COUPON)` for promotionally responsive product categories.
- Decrease `P(buy|NORMAL_BUYER, NO_COUPON)` for high-churn categories.
- Adjust transition probs to model faster/slower addiction dynamics for your category.

---

### Rationale

The RL agent learns to balance two competing effects:

| Effect                     | Short-term | Long-term          |
|----------------------------|------------|--------------------|
| Giving coupons             | ↑ conversion | ↑ CouponAddict risk |
| Withholding from addicts   | ↓ conversion | ↑ recovery toward Neutral |
| Giving to Normal Buyers    | marginal ↑  | slight downgrade risk |

A greedy "always give coupon" policy inflates the CouponAddict population over time, reducing margin per purchase. The REINFORCE agent, interacting over many episodes, learns that **selectively withholding coupons from CouponAddicts and NormalBuyers** recovers margin in the long run.

---

### BDD Testing

Feature file: `features/gradient_learning.feature`

Scenarios validated:
- State space size = 3
- Action space size = 2
- All transitions stay within `{NEUTRAL, COUPON_ADDICT, NORMAL_BUYER}`
- Reward = 0 when purchase probability is 0
- Reward = `margin` when purchase occurs without coupon
- Reward = `margin − coupon_cost` when purchase occurs with coupon
- RL policy profit > Always-Coupon baseline profit
- RL policy profit > Never-Coupon baseline profit

Run tests:
```bash
cd POC04
python -m behave
```

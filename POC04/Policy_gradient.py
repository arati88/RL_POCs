"""
POC-04: Retail Coupon Optimization using Policy Gradient (REINFORCE)
---------------------------------------------------------------------
Models a retail setting where an RL agent decides whether to issue a coupon
to a customer, balancing short-term conversion against long-term addiction risk.

States  : 3 customer segments (Neutral, CouponAddict, NormalBuyer)
Actions : 2 decisions        (NoCoupon, GiveCoupon)
Rewards : profit-based       (margin - coupon_cost if coupon given, else margin; 0 if no purchase)

Transition dynamics capture real-world coupon effects:
  - Giving coupons increases probability of drifting into CouponAddict.
  - Withholding coupons from CouponAddict gives a chance to recover toward Neutral.
  - Giving coupons to NormalBuyer carries a slight risk of downgrade.
"""

import numpy as np


# ============================================================
# STATE CONSTANTS
# ============================================================

NEUTRAL = 0         # Customer with mild interest; moderate coupon response
COUPON_ADDICT = 1   # Customer who only buys when a coupon is available
NORMAL_BUYER = 2    # Customer who buys regularly regardless of coupons

STATE_NAMES = {
    NEUTRAL: "Neutral",
    COUPON_ADDICT: "CouponAddict",
    NORMAL_BUYER: "NormalBuyer",
}

NUM_STATES = 3


# ============================================================
# ACTION CONSTANTS
# ============================================================

NO_COUPON = 0    # Do not issue a coupon
GIVE_COUPON = 1  # Issue a coupon to the customer

ACTION_NAMES = {
    NO_COUPON: "NoCoupon",
    GIVE_COUPON: "GiveCoupon",
}

NUM_ACTIONS = 2


# ============================================================
# REWARD CONFIGURATION
# ============================================================

# Profit margin earned per purchase (before coupon cost)
DEFAULT_MARGIN = 10.0

# Cost of issuing a coupon (discount + fulfilment overhead)
DEFAULT_COUPON_COST = 3.0


# ============================================================
# PURCHASE PROBABILITIES  (configurable)
# ============================================================
# P(purchase | customer_state, action)
# Defaults reflect typical retail behaviour:
#   Neutral        – moderate response; coupon roughly doubles conversion
#   CouponAddict   – rarely buys without coupon; high response with coupon
#   NormalBuyer    – buys regularly; coupon gives only a small uplift

DEFAULT_PURCHASE_PROBS = {
    (NEUTRAL,       NO_COUPON):   0.20,
    (NEUTRAL,       GIVE_COUPON): 0.45,
    (COUPON_ADDICT, NO_COUPON):   0.05,
    (COUPON_ADDICT, GIVE_COUPON): 0.60,
    (NORMAL_BUYER,  NO_COUPON):   0.65,
    (NORMAL_BUYER,  GIVE_COUPON): 0.70,
}


# ============================================================
# TRANSITION PROBABILITIES  (configurable)
# ============================================================
# T(next_state | current_state, action)
# Each entry is [P(NEUTRAL), P(COUPON_ADDICT), P(NORMAL_BUYER)]
#
# Key dynamics:
#   NoCoupon  → CouponAddict stays in addiction (slow recovery toward Neutral)
#   GiveCoupon to Neutral   → elevated drift into CouponAddict
#   GiveCoupon to NormalBuyer → small downgrade risk to Neutral/CouponAddict

DEFAULT_TRANSITION_PROBS = {
    (NEUTRAL,       NO_COUPON):   [0.70, 0.05, 0.25],  # mostly stays Neutral, small chance of becoming NormalBuyer
    (NEUTRAL,       GIVE_COUPON): [0.35, 0.45, 0.20],  # coupon risk: high drift into CouponAddict
    (COUPON_ADDICT, NO_COUPON):   [0.30, 0.60, 0.10],  # withholding helps recovery toward Neutral
    (COUPON_ADDICT, GIVE_COUPON): [0.05, 0.85, 0.10],  # coupon reinforces addiction
    (NORMAL_BUYER,  NO_COUPON):   [0.10, 0.05, 0.85],  # mostly stays NormalBuyer
    (NORMAL_BUYER,  GIVE_COUPON): [0.15, 0.15, 0.70],  # slight downgrade risk to Neutral/CouponAddict
}


# ============================================================
# RETAIL ENVIRONMENT
# ============================================================

class RetailEnv:
    """
    Custom RL environment for retail coupon optimisation.

    States   : NEUTRAL (0), COUPON_ADDICT (1), NORMAL_BUYER (2)
    Actions  : NO_COUPON (0), GIVE_COUPON (1)
    Reward   : margin - coupon_cost (if purchase + coupon given)
               margin              (if purchase + no coupon)
               0                   (if no purchase)

    Parameters
    ----------
    margin          : profit earned per successful purchase (default 10.0)
    coupon_cost     : cost of issuing a coupon (default 3.0)
    purchase_probs  : dict mapping (state, action) -> float, or None for defaults
    transition_probs: dict mapping (state, action) -> [p_neutral, p_addict, p_normal],
                      or None for defaults
    rng_seed        : optional integer seed for reproducibility
    """

    def __init__(
        self,
        margin=DEFAULT_MARGIN,
        coupon_cost=DEFAULT_COUPON_COST,
        purchase_probs=None,
        transition_probs=None,
        rng_seed=None,
    ):
        self.num_states = NUM_STATES
        self.num_actions = NUM_ACTIONS
        self.margin = margin
        self.coupon_cost = coupon_cost
        self.purchase_probs = purchase_probs if purchase_probs is not None else DEFAULT_PURCHASE_PROBS
        self.transition_probs = transition_probs if transition_probs is not None else DEFAULT_TRANSITION_PROBS
        self._rng = np.random.default_rng(rng_seed)
        self.state = NEUTRAL

    def reset(self):
        """Start a new episode from a random customer state."""
        self.state = int(self._rng.integers(0, self.num_states))
        return self.state

    def step(self, action):
        """
        Execute one step: decide on coupon, observe purchase, transition.

        Parameters
        ----------
        action : int  NO_COUPON (0) or GIVE_COUPON (1)

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool  (always False – episodic horizon set by caller)
        """
        assert action in (NO_COUPON, GIVE_COUPON), f"Invalid action: {action}"
        assert self.state in (NEUTRAL, COUPON_ADDICT, NORMAL_BUYER), f"Invalid state: {self.state}"

        # Determine purchase
        p_buy = self.purchase_probs[(self.state, action)]
        purchased = self._rng.random() < p_buy

        # Calculate reward
        if purchased:
            reward = self.margin - (self.coupon_cost if action == GIVE_COUPON else 0.0)
        else:
            reward = 0.0

        # State transition
        trans_probs = self.transition_probs[(self.state, action)]
        next_state = int(self._rng.choice(self.num_states, p=trans_probs))
        self.state = next_state

        return next_state, reward, False


# ============================================================
# POLICY GRADIENT AGENT  (REINFORCE)
# ============================================================

class PolicyGradientAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.

    Parameters
    ----------
    num_states  : int
    num_actions : int
    alpha       : float  learning rate (default 0.01)
    gamma       : float  discount factor (default 0.95)
    """

    def __init__(self, num_states, num_actions, alpha=0.01, gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma

        # Policy parameters θ, shape: (num_actions × num_states)
        self.theta = np.zeros((num_actions, num_states))
        self.num_actions = num_actions

    def softmax(self, x):
        """
        Compute softmax probabilities in a numerically stable way by
        subtracting the maximum value before exponentiation.

        Parameters
        ----------
        x : array-like  input logits

        Returns
        -------
        ndarray  probabilities summing to 1
        """
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def get_action_probs(self, state):
        """Return softmax probability vector over actions for the given state."""
        return self.softmax(self.theta[:, state])

    def select_action(self, state):
        """Sample an action stochastically from the current policy."""
        probs = self.get_action_probs(state)
        return int(np.random.choice(self.num_actions, p=probs))

    def compute_returns(self, rewards):
        """Compute discounted cumulative returns G_t for each time-step."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self, states, actions, rewards):
        """Update θ using reward-weighted policy gradient (REINFORCE)."""
        returns = np.array(self.compute_returns(rewards))

        # Normalise returns to reduce variance
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        for state, action, G in zip(states, actions, returns):
            probs = self.get_action_probs(state)

            # One-hot encoding for the taken action
            action_one_hot = np.zeros(self.num_actions)
            action_one_hot[action] = 1

            # Policy gradient: θ ← θ + α·G·(one_hot − π)
            gradient = action_one_hot - probs
            self.theta[:, state] += self.alpha * G * gradient


# ============================================================
# TRAINING
# ============================================================

def train(env, agent, episodes=3000, steps_per_episode=50):
    """Run REINFORCE training loop."""
    print("\nTraining RL Agent...\n")

    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        total_profit = 0

        for _ in range(steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            total_profit += reward

        agent.update(states, actions, rewards)

        if episode % 500 == 0:
            print(f"Episode {episode}, Profit: {total_profit:.3f}")

    print("\nTraining Complete.\n")


# ============================================================
# EVALUATION
# ============================================================

def evaluate(env, agent, steps=10000):
    """Evaluate trained policy (greedy/deterministic)."""
    state = env.reset()
    total_profit = 0
    for _ in range(steps):
        action = int(np.argmax(agent.get_action_probs(state)))
        next_state, reward, _ = env.step(action)
        total_profit += reward
        state = next_state
    return total_profit / steps


def evaluate_always_coupon(env, steps=10000):
    """Baseline: always issue a coupon."""
    state = env.reset()
    total_profit = 0
    for _ in range(steps):
        next_state, reward, _ = env.step(GIVE_COUPON)
        total_profit += reward
        state = next_state
    return total_profit / steps


def evaluate_never_coupon(env, steps=10000):
    """Baseline: never issue a coupon."""
    state = env.reset()
    total_profit = 0
    for _ in range(steps):
        next_state, reward, _ = env.step(NO_COUPON)
        total_profit += reward
        state = next_state
    return total_profit / steps


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import pandas as pd

    # Create environment and agent
    env = RetailEnv(rng_seed=42)
    agent = PolicyGradientAgent(env.num_states, env.num_actions)

    # Train agent
    train(env, agent)

    # Evaluate strategies
    rl_profit = evaluate(env, agent)
    always_profit = evaluate_always_coupon(env)
    never_profit = evaluate_never_coupon(env)

    # ============================================================
    # STRATEGY COMPARISON TABLE
    # ============================================================

    comparison_df = pd.DataFrame({
        "Strategy": [
            "RL Optimized Policy",
            "Always Give Coupon",
            "Never Give Coupon",
        ],
        "Average Profit per Step": [
            round(rl_profit, 3),
            round(always_profit, 3),
            round(never_profit, 3),
        ],
    })

    print("\n===== STRATEGY COMPARISON =====\n")
    print(comparison_df.to_string(index=False))

    best_strategy = comparison_df.loc[
        comparison_df["Average Profit per Step"].idxmax()
    ]
    print(f"\nBest Performing Strategy: {best_strategy['Strategy']}")

    # ============================================================
    # DISPLAY LEARNED POLICY
    # ============================================================

    print("\n===== LEARNED DECISION POLICY =====\n")
    for state in range(env.num_states):
        action = int(np.argmax(agent.get_action_probs(state)))
        print(f"{STATE_NAMES[state]:15s} → {ACTION_NAMES[action]}")

"""
POC-04: Retail Coupon Optimization using Policy Gradient (REINFORCE)

Models a retail setting where an RL agent decides whether to issue a
coupon to a customer, balancing short-term conversion against long-term
addiction risk.
"""

import numpy as np


# ============================================================
# STATE CONSTANTS
# ============================================================

NEUTRAL = 0
COUPON_ADDICT = 1
NORMAL_BUYER = 2

STATE_NAMES = {
    NEUTRAL: "Neutral",
    COUPON_ADDICT: "CouponAddict",
    NORMAL_BUYER: "NormalBuyer",
}

NUM_STATES = 3


# ============================================================
# ACTION CONSTANTS
# ============================================================

NO_COUPON = 0
GIVE_COUPON = 1

ACTION_NAMES = {
    NO_COUPON: "NoCoupon",
    GIVE_COUPON: "GiveCoupon",
}

NUM_ACTIONS = 2


# ============================================================
# REWARD CONFIGURATION
# ============================================================

DEFAULT_MARGIN = 10.0
DEFAULT_COUPON_COST = 3.0


# ============================================================
# PURCHASE PROBABILITIES
# ============================================================

DEFAULT_PURCHASE_PROBS = {
    (NEUTRAL, NO_COUPON): 0.20,
    (NEUTRAL, GIVE_COUPON): 0.45,
    (COUPON_ADDICT, NO_COUPON): 0.05,
    (COUPON_ADDICT, GIVE_COUPON): 0.60,
    (NORMAL_BUYER, NO_COUPON): 0.65,
    (NORMAL_BUYER, GIVE_COUPON): 0.70,
}


# ============================================================
# TRANSITION PROBABILITIES
# ============================================================

DEFAULT_TRANSITION_PROBS = {
    (NEUTRAL, NO_COUPON): [0.70, 0.05, 0.25],
    (NEUTRAL, GIVE_COUPON): [0.35, 0.45, 0.20],
    (COUPON_ADDICT, NO_COUPON): [0.30, 0.60, 0.10],
    (COUPON_ADDICT, GIVE_COUPON): [0.05, 0.85, 0.10],
    (NORMAL_BUYER, NO_COUPON): [0.10, 0.05, 0.85],
    (NORMAL_BUYER, GIVE_COUPON): [0.15, 0.15, 0.70],
}


# ============================================================
# RETAIL ENVIRONMENT
# ============================================================

class RetailEnv:
    """Custom RL environment for retail coupon optimisation."""

    def __init__(
        self,
        margin: float = DEFAULT_MARGIN,
        coupon_cost: float = DEFAULT_COUPON_COST,
        purchase_probs=None,
        transition_probs=None,
        rng_seed: int | None = None,
    ) -> None:

        self.num_states = NUM_STATES
        self.num_actions = NUM_ACTIONS
        self.margin = margin
        self.coupon_cost = coupon_cost

        self.purchase_probs = (
            purchase_probs
            if purchase_probs is not None
            else DEFAULT_PURCHASE_PROBS
        )

        self.transition_probs = (
            transition_probs
            if transition_probs is not None
            else DEFAULT_TRANSITION_PROBS
        )

        self._rng = np.random.default_rng(rng_seed)
        self.state = NEUTRAL

    def reset(self) -> int:
        """Start a new episode from a random customer state."""
        self.state = int(self._rng.integers(0, self.num_states))
        return self.state

    def step(self, action: int):
        """Execute one environment step."""

        if action not in (NO_COUPON, GIVE_COUPON):
            raise ValueError(f"Invalid action: {action}")

        p_buy = self.purchase_probs[(self.state, action)]
        purchased = self._rng.random() < p_buy

        if purchased:
            reward = self.margin - (
                self.coupon_cost if action == GIVE_COUPON else 0.0
            )
        else:
            reward = 0.0

        trans_probs = self.transition_probs[(self.state, action)]
        next_state = int(
            self._rng.choice(self.num_states, p=trans_probs)
        )

        self.state = next_state
        return next_state, reward, False


# ============================================================
# POLICY GRADIENT AGENT (REINFORCE)
# ============================================================

class PolicyGradientAgent:
    """REINFORCE (Monte Carlo Policy Gradient) agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.01,
        gamma: float = 0.95,
    ) -> None:

        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        self.theta = np.zeros((num_actions, num_states))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute numerically stable softmax."""
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def get_action_probs(self, state: int) -> np.ndarray:
        """Return action probabilities for given state."""
        return self.softmax(self.theta[:, state])

    def select_action(self, state: int) -> int:
        """Sample action from policy."""
        probs = self.get_action_probs(state)
        return int(np.random.choice(self.num_actions, p=probs))

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        g_return = 0

        for reward in reversed(rewards):
            g_return = reward + self.gamma * g_return
            returns.insert(0, g_return)

        return returns

    def update(self, states, actions, rewards) -> None:
        """Update policy parameters using REINFORCE."""
        returns = np.array(self.compute_returns(rewards))

        returns = (
            returns - np.mean(returns)
        ) / (np.std(returns) + 1e-8)

        for state, action, g_value in zip(states, actions, returns):
            probs = self.get_action_probs(state)

            action_one_hot = np.zeros(self.num_actions)
            action_one_hot[action] = 1.0

            gradient = action_one_hot - probs
            self.theta[:, state] += self.alpha * g_value * gradient


# ============================================================
# TRAINING
# ============================================================

def train(env, agent, episodes: int = 3000,
          steps_per_episode: int = 50) -> None:
    """Run REINFORCE training loop."""

    print("\nTraining RL Agent...\n")

    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        total_profit = 0.0

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

def evaluate(env, agent, steps: int = 10000) -> float:
    """Evaluate trained policy (greedy)."""
    state = env.reset()
    total_profit = 0.0

    for _ in range(steps):
        action = int(np.argmax(agent.get_action_probs(state)))
        state, reward, _ = env.step(action)
        total_profit += reward

    return total_profit / steps


def evaluate_always_coupon(env, steps: int = 10000) -> float:
    """Baseline: always give coupon."""
    state = env.reset()
    total_profit = 0.0

    for _ in range(steps):
        state, reward, _ = env.step(GIVE_COUPON)
        total_profit += reward

    return total_profit / steps


def evaluate_never_coupon(env, steps: int = 10000) -> float:
    """Baseline: never give coupon."""
    state = env.reset()
    total_profit = 0.0

    for _ in range(steps):
        state, reward, _ = env.step(NO_COUPON)
        total_profit += reward

    return total_profit / steps


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import pandas as pd

    env = RetailEnv(rng_seed=42)
    agent = PolicyGradientAgent(
        env.num_states,
        env.num_actions,
    )

    train(env, agent)

    rl_profit = evaluate(env, agent)
    always_profit = evaluate_always_coupon(env)
    never_profit = evaluate_never_coupon(env)

    comparison_df = pd.DataFrame(
        {
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
        }
    )

    print("\n===== STRATEGY COMPARISON =====\n")
    print(comparison_df.to_string(index=False))

    best_strategy = comparison_df.loc[
        comparison_df["Average Profit per Step"].idxmax()
    ]

    print(
        f"\nBest Performing Strategy: "
        f"{best_strategy['Strategy']}"
    )

    print("\n===== LEARNED DECISION POLICY =====\n")

    for state in range(env.num_states):
        action = int(np.argmax(agent.get_action_probs(state)))
        print(
            f"{STATE_NAMES[state]:15s} → "
            f"{ACTION_NAMES[action]}"
        )
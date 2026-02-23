"""
POC-04: Retail Coupon Optimization using Policy Gradient
--------------------------------------------------------
- Generates synthetic data (100 rows)
- Saves CSV
- Builds RL environment
- Trains Policy Gradient agent
- Compares against baseline strategies
- Displays comparison table
"""

import pandas as pd
import numpy as np
import os


# ============================================================
# STEP 1: GENERATE SYNTHETIC DATA
# ============================================================

def generate_data(num_rows=100):

    np.random.seed(42)  #Fix random seed so results are reproducible
    data = []    # Fix random seed so results are reproducible

    for i in range(num_rows):

        # Randomly assign customer type
        # 30% sensitive, 40% normal, 30% neutral
        customer_type = np.random.choice(
            ["coupon_sensitive", "normal", "neutral"],
            p=[0.3, 0.4, 0.3]
        )

        # Random base purchase amount between 200 and 1000
        base_purchase = np.random.randint(200, 1000)

        # Define how much spending increases if coupon is given
        if customer_type == "coupon_sensitive":
            coupon_multiplier = 2.0    #Strong reaction
        elif customer_type == "normal":
            coupon_multiplier = 1.2    #Mild reaction
        else:
            coupon_multiplier = 1.3   #Moderate reaction

        discount_rate = 0.2   #20% Discount 

        # Profit calculations
        # Profit if NO coupon is given
        profit_no_coupon = base_purchase * 0.3   #30% margin

        # Profit if coupon is given
        profit_coupon = (base_purchase * coupon_multiplier) * 0.3 - (base_purchase * discount_rate)

        #Store row
        data.append([
            i,
            customer_type,
            base_purchase,
            round(profit_no_coupon, 3),
            round(profit_coupon, 3)
        ])

    #Convert to dataframe
    df = pd.DataFrame(data, columns=[
        "customer_id",
        "customer_type",
        "base_purchase",
        "profit_no_coupon",
        "profit_with_coupon"
    ])

    #Save CSV
    df.to_csv("retail_100_rows.csv", index=False)
    print("Dataset Generated (100 rows).")

    return df


# ============================================================
# STEP 2: RETAIL ENVIRONMENT
# ============================================================

class RetailEnv:

    def __init__(self, df):
        self.df = df                #Store data set
        self.num_states = 3         #3 Customer types
        self.num_actions = 2        # 0 = no coupon, 1 = coupon


        #Convert customer type to numeric state
        self.type_map = {
            "coupon_sensitive": 0,
            "normal": 1,
            "neutral": 2
        }

    def reset(self):
        # Pick random customer as starting state
        self.current_index = np.random.randint(0, len(self.df))
        row = self.df.iloc[self.current_index]
        return self.type_map[row["customer_type"]]

    def step(self, action):

        # Get current customer
        row = self.df.iloc[self.current_index]

        # Reward depends on action chosen
        reward = row["profit_with_coupon"] if action == 1 else row["profit_no_coupon"]

        #Move to next random customer
        self.current_index = np.random.randint(0, len(self.df))
        next_row = self.df.iloc[self.current_index]
        next_state = self.type_map[next_row["customer_type"]]

        return next_state, reward, False


# ============================================================
# STEP 3: POLICY GRADIENT AGENT
# ============================================================

class PolicyGradientAgent:

    def __init__(self, num_states, num_actions, alpha=0.01, gamma=0.95):

        self.alpha = alpha   # Learning rate
        self.gamma = gamma   # Discount factor

        # Theta = policy parameters
        # Shape: (actions x states)
        self.theta = np.zeros((num_actions, num_states))
        self.num_actions = num_actions

    def softmax(self, x):

        #Convert raw values into probabilities
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def get_action_probs(self, state):
        # Get probability of each action for given state
        return self.softmax(self.theta[:, state])

    def select_action(self, state):
        # Choose action based on probabilities (STOCHASTIC)
        probs = self.get_action_probs(state)
        return np.random.choice(self.num_actions, p=probs)

    def compute_returns(self, rewards):
        # Compute discounted cumulative rewards
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self, states, actions, rewards):
        
        # Compute discounted returns
        returns = self.compute_returns(rewards)
        
        #Normalize returns (stabilizes training)
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        #Update policy parameters
        for state, action, G in zip(states, actions, returns):

            probs = self.get_action_probs(state)

            #One-hot encoding for selected actions
            action_one_hot = np.zeros(self.num_actions)
            action_one_hot[action] = 1

            #Policy gradient formula
            gradient = action_one_hot - probs

            #Reward weighted gradient update
            self.theta[:, state] += self.alpha * G * gradient


# ============================================================
# STEP 4: TRAINING
# ============================================================

def train(env, agent, episodes=3000, steps_per_episode=50):

    print("\nTraining RL Agent...\n")

    for episode in range(episodes):

        state = env.reset()
        states, actions, rewards = [], [], []
        total_profit = 0

        # Run one episode
        for _ in range(steps_per_episode):

            action = agent.select_action(state)
            next_state, reward, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            total_profit += reward

        #Update policy after episode
        agent.update(states, actions, rewards)

        if episode % 500 == 0:
            print(f"Episode {episode}, Profit: {total_profit:.3f}")

    print("\nTraining Complete.\n")


# ============================================================
# STEP 5: EVALUATION
# ============================================================

def evaluate(env, agent, steps=10000):

    #Deterministic evaluation ( choose best action)
    state = env.reset()
    total_profit = 0

    for _ in range(steps):

        action = np.argmax(agent.get_action_probs(state))
        next_state, reward, _ = env.step(action)

        total_profit += reward
        state = next_state

    return total_profit / steps


def evaluate_always_coupon(env, steps=10000):

    state = env.reset()
    total_profit = 0

    for _ in range(steps):
        next_state, reward, _ = env.step(1)     # Always coupon
        total_profit += reward
        state = next_state

    return total_profit / steps


def evaluate_never_coupon(env, steps=10000):

    state = env.reset()
    total_profit = 0

    for _ in range(steps):
        next_state, reward, _ = env.step(0)    # Never coupon
        total_profit += reward
        state = next_state

    return total_profit / steps


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Load dataset if exists, else generate new
    if os.path.exists("retail_100_rows.csv"):
        df = pd.read_csv("retail_100_rows.csv")
        print("Loaded existing dataset.")
    else:
        df = generate_data()

    # Create environment and agent
    env = RetailEnv(df)
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
            "Never Give Coupon"
        ],
        "Average Profit per Step": [
            round(rl_profit, 3),
            round(always_profit, 3),
            round(never_profit, 3)
        ]
    })

    print("\n===== STRATEGY COMPARISON =====\n")
    print(comparison_df.to_string(index=False))

    # Identify best strategy
    best_strategy = comparison_df.loc[
        comparison_df["Average Profit per Step"].idxmax()
    ]

    print("\nBest Performing Strategy:")
    print(f"{best_strategy['Strategy']}")

    # ====================================
    # DISPLAY LEARNED POLICY
    # ====================================

    print("\n===== LEARNED DECISION POLICY =====\n")

    customer_names = {
        0: "Coupon Sensitive Customer",
        1: "Normal Customer",
        2: "Neutral Customer"
    }

    for state in range(env.num_states):
        action = np.argmax(agent.get_action_probs(state))
        decision = "Give Coupon" if action == 1 else "No Coupon"
        print(f"{customer_names[state]} → {decision}")
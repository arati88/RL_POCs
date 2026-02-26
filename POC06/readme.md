# POC-06: REINFORCE Trading Policy - Nueral Networks as policy approximators

# Overview
This POC demonstrates a proof-of-concept Reinforcement Learning agent for stock trading using policy gradient methods (REINFORCE) with a value baseline. The agent learns when to BUY, SELL, or HOLD based on a set of continuous market indicators.

Key Features:
    Neural policy network π(a|s) to choose actions.
    Value network baseline to reduce gradient variance (Actor-Critic style).
    Handles continuous state space — Q-tables infeasible.
    Action masking ensures only valid trades are executed.
    Cooldown constraints prevent over-trading.
    Reward shaping for profit maximization and patience.
    Deterministic evaluation via argmax of policy logits.

# How It Works

Environment (StockEnv)
    Simulates a trending GBM market.
    State: [price_change, RSI, MA_ratio, position, unrealized PnL].
    Actions: HOLD=0, BUY=1, SELL=2.
    Cooldown prevents immediate repeated trading.
    Reward shaped to encourage profitable trades and penalize losses.

Policy Network (PolicyNetwork)
    Inputs: 5 continuous state features.
    Outputs: logits for 3 actions.
    Uses LayerNorm + ReLU in 2 hidden layers.
    Action masking prevents invalid trades.
    Sampling from Categorical(logits) during training.

Value Network (ValueNetwork)
    Estimates V(s) — expected future reward from state s.
    Used as baseline to reduce variance in REINFORCE updates.
    
Training (REINFORCETrainer)
    Computes discounted returns for each trajectory.
    Calculates advantage = G_t - V(s).
    Updates policy and value networks separately.
    Applies entropy bonus for exploration.

Evaluation
    Uses deterministic policy: action = argmax π(a|s).
    Measures final portfolio, alpha vs Buy & Hold, trades executed, average holding duration.

# Example Output
Ep 300/300  |  Avg P&L: +1.4%  |  Win: 58%  |  H=93% B=3% S=3%  |  Entropy: 0.126
Best P&L        : +22.0%
Avg P&L last 50 : +1.4%
Win rate last 50: 58%

EVALUATION  (deterministic — argmax of π(a|s))
Step  10 | BUY  @ $94.07  | π: H=0.43 B=0.57 S=0.00  | $  9,990.50
Step  33 | SELL @ $99.34  | π: H=0.43 B=0.00 S=0.57  | $ 10,512.50  held 23 steps

Policy NN  : $10,463.91  (+4.64%)
Trades     : 3 BUY + 3 SELL
Avg hold   : 28 steps per trade



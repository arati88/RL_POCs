"""
=============================================================================
 STOCK TRADING AGENT — RL with Neural Network Policy 
 Framework : PyTorch
 Algorithm : REINFORCE with Value Baseline (Actor–Critic style)
 
 Key Features
 ------------
 • Neural policy approximator (π(a|s))
 • Value network baseline to reduce variance
 • Proper gradient accumulation over trajectory
 • Action masking (invalid trades prevented)
 • Cooldown constraint to avoid rapid churn
 • Reward shaping for patience & loss control
 • Deterministic evaluation (argmax policy)

 Purpose
 -------
 Proof-of-concept demonstrating that policy gradient methods
 can learn structured trading behavior under proper reward design.
=============================================================================
"""

import os, random, numpy as np

from collections import deque       ## Fast append + pop operations.Tracking last 50 episode returns

import torch
import torch.nn as nn               #Neural network module.
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# =============================================================================
# ENVIRONMENT
# =============================================================================

class StockEnv:  # defining a custom reinforcement learning environment.
    """
    Trending GBM market. Agent must learn WHEN to buy and WHEN to sell
    based on 5 continuous indicators — not just "sell after cooldown."

    State (5 continuous features — Q-table infeasible):
      [0] price_change  : recent log return (short-term momentum)
      [1] rsi           : RSI(14) momentum (overbought/oversold)
      [2] ma_ratio      : price vs 20-day MA (trend signal)
      [3] position      : 1 = holding stock, 0 = no position
      [4] unrealized_pnl: current % gain/loss if holding
    """
    STATE_DIM = 5   # State space is 5-dimensional continuous vector
    N_ACTIONS = 3   # Action space is discrete with 3 actions: 0=HOLD  1=BUY  2=SELL

    def __init__(self, initial_cash=10_000, steps=120): # Each episode is 120 timesteps of simulated market.
        """
        initial_cash : Starting capital
        steps        : Episode length (market timesteps)
        """
        self.initial_cash = initial_cash
        self.steps        = steps
        self.prices       = deque(maxlen=30)   # Stores last 30 prices. (used for RSI & moving average)
        self.reset()            # Environment automatically initializes.

    # ==========================================================
    # MARKET GENERATOR (Trending GBM with Regime Switching)
    # ==========================================================
    def _next_price(self):   
        self.trend_steps += 1   # Track how long current trend lasts.

        # When trend duration expires → flip direction
        # Uptrend becomes downtrend and vice versa
        if self.trend_steps >= self.trend_duration: 
            self.trend          = -self.trend     
            self.trend_duration = np.random.randint(25, 60)
            self.trend_steps    = 0

        # Geometric Brownian Motion formula:
        # return = drift + volatility shock
        ret = (self.trend - 0.5 * self.sigma**2) + self.sigma * np.random.randn()  

        # Exponential update (ensures price stays positive)
        return max(self.price * np.exp(ret), 1.0)   
    
    # ==========================================================
    # RSI Indicator (Momentum strength)
    # ==========================================================

    def _rsi(self, w=14):
        """
        RSI normalized between 0 and 1
        Near 1 → Overbought
        Near 0 → Oversold
        """
        p = list(self.prices)
        if len(p) < w + 1: return 0.5   # Neutral if insufficient history

        ch = np.diff(p[-(w+1):])

        g  = ch[ch > 0].mean() if (ch > 0).any() else 0.0
        l  = -ch[ch < 0].mean() if (ch < 0).any() else 1e-9
        return float(np.clip(1 - 1 / (1 + g / (l + 1e-9)), 0, 1))

    # ==========================================================
    # Moving Average Ratio
    # ==========================================================
    def _ma_ratio(self, w=20):
        """
        Measures how far price is from 20-day MA
        Positive → Above MA (uptrend)
        Negative → Below MA (downtrend)
        """

        p = list(self.prices)
        if len(p) < w: return 0.0

        return float(np.clip((self.price / np.mean(p[-w:])) - 1, -0.15, 0.15) / 0.15)
    # ==========================================================
    # STATE REPRESENTATION
    # ==========================================================
    def _state(self):
        p   = list(self.prices)

        # Short-term log return
        ret = np.log(p[-1] / (p[-2] + 1e-9)) if len(p) >= 2 else 0.0

        # Unrealized PnL if holding
        unr = (self.price - self.buy_price) / self.buy_price if self.buy_price else 0.0
        return np.array([
            np.clip(ret, -0.1, 0.1) / 0.1,      # normalized return
            self._rsi(),                        # RSI (0–1)
            self._ma_ratio(),                   # MA signal
            1.0 if self.shares > 0 else 0.0,    # position flag
            np.clip(unr, -1.0, 1.0),            # unrealized gain/loss
        ], dtype=np.float32)

    # ==========================================================
    # VALID ACTIONS
    # ==========================================================
    def valid_actions(self):
        """
        Cooldown prevents over-trading.
        If cooldown active → only HOLD allowed.
        """
        if self.cooldown > 0: return [0]

        # If holding → can SELL
        # If flat → can BUY
        return [0, 2] if self.shares > 0 else [0, 1]
    
    # ==========================================================
    # RESET ENVIRONMENT
    # ==========================================================
    def reset(self):
        self.cash           = float(self.initial_cash)
        self.shares         = 0.0
        self.buy_price      = None
        self.buy_step       = None
        self.cooldown       = 0
        self.step_n         = 0

         # Initial price and volatility
        self.price          = 100.0
        self.sigma          = 0.008

        # Random starting trend
        self.trend          = np.random.choice([-0.003, 0.003])
        self.trend_duration = np.random.randint(25, 60)
        self.trend_steps    = 0
        self.prices.clear()

        # Warm up indicators
        for _ in range(25):                          # warm up indicators
            self.price = self._next_price()
            self.prices.append(self.price)
        self.portfolio_vals = [self.initial_cash]
        return self._state()

    # ==========================================================
    # STEP FUNCTION (Core RL Interaction)
    # ==========================================================
    def step(self, action):
        old_price  = self.price
        # Generate next market price
        self.price = self._next_price()
        self.prices.append(self.price)

        valid = self.valid_actions()
        # Enforce valid actions
        if action not in valid:
            action = 0

        traded, ttype = False, "HOLD"
        reward        = 0.0

        # ------------------------------------------------------
        # BUY ACTION
        # ------------------------------------------------------
        if action == 1 and self.cash > 1:            
            
            # Invest 95% of cash (keep buffer)
            spend          = self.cash * 0.95
            self.shares    = spend / self.price

            # Transaction cost included
            self.cash     -= spend * 1.001
            self.buy_price = self.price
            self.buy_step  = self.step_n

            # Cooldown prevents rapid trading
            self.cooldown  = 10                      # FIX: 3 → 10 steps
            traded, ttype  = True, "BUY"

            # Entry gives no reward
            reward         = 0.0                     

        # ------------------------------------------------------
        # SELL ACTION
        # ------------------------------------------------------
        elif action == 2 and self.shares > 0:        # SELL
            proceeds      = self.shares * self.price * 0.999
            realized_pnl  = proceeds - (self.shares * self.buy_price)
            hold_duration = self.step_n - self.buy_step + 1

            self.cash    += proceeds
            self.shares   = 0.0
            self.buy_price = None
            self.buy_step  = None
            self.cooldown  = 10
            traded, ttype  = True, "SELL"

            # Reward = realized P&L, scaled up for longer holds
            # This teaches: "hold profitable positions longer"
            duration_bonus = min(hold_duration / 20, 2.0)
            reward = (realized_pnl / self.initial_cash * 100) * duration_bonus

        # ------------------------------------------------------
        # HOLD ACTION (While in position)
        # ------------------------------------------------------
        elif action == 0 and self.shares > 0:        # HOLD with position
            price_move = (self.price - old_price) / old_price

            # Reward price appreciation, penalize holding losses
            reward = price_move * 20

            # Extra penalty for sitting on an unrealized loss > 2%
            unr = (self.price - self.buy_price) / self.buy_price
            if unr < -0.02:
                reward -= 0.1                        # nudge: cut your losers

        # Clip extreme rewards
        reward = float(np.clip(reward, -5.0, 5.0))

        # Reduce cooldown
        if self.cooldown > 0 and not traded:
            self.cooldown -= 1

        # Portfolio value update
        portfolio = self.cash + self.shares * self.price
        self.portfolio_vals.append(portfolio)
        self.step_n += 1
        done = self.step_n >= self.steps

        return self._state(), reward, done, {
            'portfolio': portfolio, 'price': self.price,
            'trade': ttype, 'traded': traded, 'step': self.step_n
        }


# =============================================================================
# NETWORKS
# =============================================================================

class PolicyNetwork(nn.Module):
    """π(a|s) — the policy approximator. NN IS the policy.
    INPUT:
        state (5 continuous features)

    OUTPUT:
        logits for 3 actions (HOLD, BUY, SELL)

    NOTE:
        Logits are converted to probabilities internally
        using a Categorical distribution.
    """
    def __init__(self, state_dim=5, hidden=128, n_actions=3):
        super().__init__()

        # Number of possible actions
        self.n_actions = n_actions

        # Network Architecture
        # ------------------------------------------------------
        # state → hidden → hidden → action logits
        #
        # LayerNorm stabilizes training (important in RL)
        # ReLU adds non-linearity
        #
        # Why 2 hidden layers?
        #   To capture complex non-linear relationships between:
        #   - RSI
        #   - MA ratio
        #   - price momentum
        #   - unrealized PnL
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),    # outputs raw action scores (logits)
        )

        # Weight Initialization
        # ------------------------------------------------------
        # Xavier initialization keeps gradients stable.
        # Important in RL because training signals are noisy.

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    # Forward Pass
    def forward(self, x): 
        """
        Takes state tensor → outputs action logits.

        No softmax here!
        We pass logits directly to Categorical distribution,
        which handles softmax internally.
        """
        return self.net(x)

    # Action Selection
    # ----------------------------------------------------------
    def act(self, state, valid_actions):
        """
        Selects an action based on current policy.

        Steps:
        1. Convert state to tensor
        2. Get action logits from network
        3. Mask invalid actions (cooldown logic)
        4. Create probability distribution
        5. Sample action
        6. Return action, log probability, entropy

        Returns:
            action      → chosen action (int)
            log_prob    → log π(a|s) (used in policy gradient update)
            entropy     → randomness measure (used for exploration bonus)
        """

        # Convert state (numpy array) → PyTorch tensor
        s      = torch.FloatTensor(state)

        # Get raw action scores
        logits = self.forward(s)

        # ACTION MASKING
        # 
        # Some actions may be invalid (e.g., SELL when not holding)
        # We assign -inf to invalid actions so probability = 0
        mask   = torch.full((self.n_actions,), float('-inf'))


        for a in valid_actions: mask[a] = 0.0

        # Create categorical distribution over actions
        dist   = Categorical(logits=logits + mask)

        # Sample action from probability distribution
        action = dist.sample()

        # Return:
        # - chosen action
        # - log probability (used in policy gradient loss)
        # - entropy (encourages exploration)
        return action.item(), dist.log_prob(action), dist.entropy()


class ValueNetwork(nn.Module):
    """""
    V(s) — State Value Function Approximator

    This network estimates:

        V(s) = Expected future total reward from state s

    Why do we need this?

    In REINFORCE:
        Policy gradient uses total return Gt

    But returns are very noisy → high variance gradients.

    So we subtract a baseline:
        Advantage = Gt - V(s)

    This reduces variance while keeping gradient unbiased.

    This network is called the "Critic".
    """""
    def __init__(self, state_dim=5, hidden=128):
        super().__init__()

        # Network Architecture
        # ------------------------------------------------------
        # Input  : state (5 continuous features)
        # Hidden : 2 layers with ReLU
        # Output : Single scalar value → V(s)
        #
        # Unlike PolicyNetwork:
        # - No LayerNorm (simpler regression task)
        # - Output dimension = 1
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Weight Initialization
        # ------------------------------------------------------
        # Xavier initialization:
        # Keeps variance stable across layers
        # Important for stable RL training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    # Forward Pass
    def forward(self, x): 
        """
        Input:
            x → state tensor

        Output:
            Scalar value V(s)

        squeeze(-1):
            Removes last dimension so output shape is:
            [batch_size] instead of [batch_size, 1]

        This makes loss calculation easier.
        """
        return self.net(x).squeeze(-1)


# =============================================================================
# REINFORCE TRAINER
# =============================================================================

class REINFORCETrainer:
    """
    This class performs policy updates using:

        REINFORCE with baseline (Value Network)

    Actor  → PolicyNetwork  (learns π(a|s))
    Critic → ValueNetwork   (learns V(s))

    Policy update uses Advantage:
        Advantage = Return - V(s)
    """
    def __init__(self, policy, value,
                 policy_lr=3e-4, value_lr=8e-4,
                 gamma=0.99, entropy_coeff=0.03):
        
        # Actor (Policy Network)
        self.policy        = policy

        # Critic (Value Network)
        self.value         = value

        # Discount factor (how much future rewards matter)
        self.gamma         = gamma

        # Entropy coefficient (encourages exploration)
        self.entropy_coeff = entropy_coeff

        # Separate Optimizers
        # ------------------------------------------------------
        # Policy and value are trained separately
        # They have different learning rates
        self.policy_opt = optim.Adam(policy.parameters(), lr=policy_lr)
        self.value_opt  = optim.Adam(value.parameters(),  lr=value_lr)

        # Learning rate scheduler
        # Every 100 updates → LR becomes half
        self.policy_sched = optim.lr_scheduler.StepLR(
            self.policy_opt, step_size=100, gamma=0.5)
        self.value_sched  = optim.lr_scheduler.StepLR(
            self.value_opt,  step_size=100, gamma=0.5)

        # For tracking training behavior
        self.entropy_log   = []
        self.policy_losses = []

    # Compute Discounted Returns
    # ==========================================================
    def _returns(self, rewards):
        """
        Computes:

            G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...

        Done backward because:
            Each return depends on future rewards.

        Also normalizes returns to stabilize training.
        """

        G, returns = 0.0, []

        # Compute from end to beginning
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        t = torch.FloatTensor(returns)
        # Normalize returns (very important in RL)
        if t.std() > 1e-6:
            t = (t - t.mean()) / (t.std() + 1e-8)
        return t

    # Update Policy and Value Networks
    # ==========================================================
    def update(self, states, rewards, log_probs, entropies):

        # 1 Compute discounted returns
        returns    = self._returns(rewards)

        # Convert states to tensor
        states_t   = torch.FloatTensor(np.array(states))

        # Get predicted state values V(s)
        values     = self.value(states_t)

        # 2 Compute Advantage
        # ------------------------------------------
        # Advantage = Actual return - Predicted value
        #
        # detach() prevents policy gradient
        # from flowing into value network
        advantages = returns - values.detach()

        # Normalize advantages for stability
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack saved log probabilities and entropies
        log_probs_t  = torch.stack(log_probs)
        entropies_t  = torch.stack(entropies)

        # Track entropy for analysis
        self.entropy_log.append(entropies_t.mean().item())

        # 3 POLICY UPDATE (Actor)
        # ======================================================
        #
        # Policy gradient objective:
        #
        #     Loss = - log π(a|s) * Advantage
        #
        # If advantage positive:
        #     Increase probability
        #
        # If advantage negative:
        #     Decrease probability

        policy_loss = -(log_probs_t * advantages).mean()
        # Entropy bonus encourages exploration
        # High entropy = more randomness
        entropy_loss = -self.entropy_coeff * entropies_t.mean()

        # Zero gradients
        self.policy_opt.zero_grad()

        # Backprop through policy network
        (policy_loss + entropy_loss).backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

        # Update actor parameters
        self.policy_opt.step()

        # 4 VALUE UPDATE (Critic)
        # ======================================================
        #
        # Critic learns to predict returns:
        #
        #     Loss = MSE(V(s), G_t)
        value_loss = F.mse_loss(values, returns)
        self.value_opt.zero_grad()

        # Backprop through value network
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_opt.step()

        # Learning rate scheduling
        self.policy_sched.step()
        self.value_sched.step()

        # Track policy loss
        self.policy_losses.append(policy_loss.item())


# =============================================================================
# TRAINING
# =============================================================================

def train(num_episodes=300, initial_cash=10_000):

    """
    Main training loop.

    This function:
    1 Creates environment
    2 Initializes Actor & Critic networks
    3 Runs episodes
    4 Collects trajectories
    5 Updates networks after each episode
    6 Tracks performance metrics

    We are training using:
        REINFORCE + Baseline (Actor–Critic style)
    """

    # Training Configuration Summary
    print("=" * 66)
    print("  REINFORCE + Baseline — Neural Network as Policy Approximator")
    print("=" * 66)
    print(f"  State      : 5 continuous features (RSI, MA, return, pos, unr)")
    print(f"  Q-table    : 10^5 × 3 = 300K rows  →  infeasible")
    print(f"  Policy NN  : π(a|s) → [P(HOLD), P(BUY), P(SELL)] directly")
    print(f"  Market     : Trending GBM  |  Cooldown: 10 steps")
    print(f"  Reward     : realized P&L × duration bonus on SELL")
    print(f"  Episodes   : {num_episodes}  |  Steps/ep: 120  |  γ=0.99")
    print("=" * 66)

    # 1 Create Environment
    env     = StockEnv(initial_cash=initial_cash, steps=120)   # 300→120

    # 2 Initialize Networks
    #
    # Policy (Actor) → outputs action probabilities
    policy  = PolicyNetwork(state_dim=StockEnv.STATE_DIM, hidden=128,  # 256→128
                            n_actions=StockEnv.N_ACTIONS)
    
    # Value (Critic) → estimates V(s)
    value   = ValueNetwork(state_dim=StockEnv.STATE_DIM, hidden=64)    # 128→64

    # Trainer handles policy + value updates
    trainer = REINFORCETrainer(policy, value, policy_lr=3e-4, value_lr=8e-4,
                               gamma=0.99, entropy_coeff=0.03)

    # Trainer handles policy + value updates
    # Total reward per episode, % P&L per episode, Action distribution per episode
    rewards_log, pnl_log, action_dist_log = [], [], []

    # 3 EPISODE LOOP
    for ep in range(1, num_episodes + 1):

        state = env.reset()     # Reset market
        done  = False

        # Store full trajectory for this episode
        ep_states, ep_rewards      = [], []
        ep_log_probs, ep_entropies = [], []

        # Track how often each action is used
        action_counts = {0: 0, 1: 0, 2: 0}

        # 4 STEP LOOP (Interaction Phase)
        while not done:

            # Get valid actions (cooldown logic applied)
            valid = env.valid_actions()

            # Actor chooses action
            action, log_prob, entropy = policy.act(state, valid)

            # Environment executes action
            next_state, reward, done, info = env.step(action)

            # Store trajectory data
            ep_states.append(state)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_entropies.append(entropy)

            # Track action frequency
            action_counts[action] += 1
            
            # Move to next state
            state = next_state
        
        # 5 UPDATE NETWORKS (Learning Phase)
        trainer.update(ep_states, ep_rewards, ep_log_probs, ep_entropies)

        # 6 PERFORMANCE METRICS
        # Final portfolio value
        final = env.portfolio_vals[-1]

        # Percentage profit/loss
        pnl   = (final - initial_cash) / initial_cash * 100
        rewards_log.append(sum(ep_rewards))
        pnl_log.append(pnl)
        action_dist_log.append(action_counts)


        # Print progress every 50 episodes
        if ep % 50 == 0 or ep == 1:
            avg_pnl  = np.mean(pnl_log[-50:])
            win_rate = np.mean([1 if p > 0 else 0 for p in pnl_log[-50:]]) * 100

            # Action percentages
            h_pct    = action_counts[0] / env.steps * 100
            b_pct    = action_counts[1] / env.steps * 100
            s_pct    = action_counts[2] / env.steps * 100

            # Average entropy (exploration measure)
            ent      = np.mean(trainer.entropy_log[-50:])
            print(f"  Ep {ep:>3}/{num_episodes}  |  "
                  f"Avg P&L: {avg_pnl:+5.1f}%  |  "
                  f"Win: {win_rate:.0f}%  |  "
                  f"H={h_pct:.0f}% B={b_pct:.0f}% S={s_pct:.0f}%  |  "
                  f"Entropy: {ent:.3f}")

    # 7 Final Summary
    print("=" * 66)
    print(f"  Best P&L        : {max(pnl_log):+.1f}%")
    print(f"  Avg P&L last 50 : {np.mean(pnl_log[-50:]):+.1f}%")
    print(f"  Win rate last 50: {np.mean([1 if p>0 else 0 for p in pnl_log[-50:]])*100:.0f}%")
    print("=" * 66)
    return policy, value, trainer, rewards_log, pnl_log, action_dist_log


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(policy: PolicyNetwork, initial_cash=10_000):

    """
    Evaluate trained policy in deterministic mode.

    During training:
        Action ~ Categorical distribution (sampling)

    During evaluation:
        Action = argmax π(a|s)
        → Choose highest probability action
        → No randomness

    This simulates real-world deployment of the strategy.
    """


    print("\n  EVALUATION  (deterministic — argmax of π(a|s))")
    print("  " + "-" * 54)

    # 1 Create Fresh Environment
    env       = StockEnv(initial_cash=initial_cash, steps=120)
    state     = env.reset()
    done      = False

    # Track portfolio and prices for comparison
    portfolio = [initial_cash]
    prices    = [env.price]

    # Trade tracking
    buys, sells, hold_durations = [], [], []
    buy_step_log = None  # Track when a trade was opened

    # 2 Switch Policy to Evaluation Mode
    # Disables dropout / normalization effects if present
    policy.eval()

    # No gradient tracking needed during evaluation
    with torch.no_grad():
        while not done:

            # Get valid actions (cooldown logic applied)
            valid  = env.valid_actions()

            # Convert state to tensor
            s  = torch.FloatTensor(state)

            # Forward pass → logits
            logits = policy(s)
            mask   = torch.full((StockEnv.N_ACTIONS,), float('-inf'))
            for a in valid: mask[a] = 0.0

            # Convert logits → probabilities
            probs  = F.softmax(logits + mask, dim=-1)

            # Deterministic action selection
            action = int(probs.argmax())

            # Take step in environment
            state, _, done, info = env.step(action)

            # Track portfolio and price
            portfolio.append(info['portfolio'])
            prices.append(info['price'])

            # Trade Logging
            if info['traded']:
                tag = "BUY " if info['trade'] == "BUY" else "SELL"
                pnl_str = ""

                # If SELL → compute holding duration
                if info['trade'] == "SELL" and buy_step_log is not None:
                    dur      = info['step'] - buy_step_log
                    hold_durations.append(dur)
                    pnl_str  = f"  held {dur} steps"

                    # Print trade details
                print(f"  Step {info['step']:>3} | {tag} @ ${info['price']:.2f}"
                      f"  | π: H={probs[0]:.2f} B={probs[1]:.2f} S={probs[2]:.2f}"
                      f"  | ${info['portfolio']:>10,.2f}{pnl_str}")
                
                # Track BUY/SELL steps
                if info['trade'] == 'BUY':
                    buys.append(info['step'])
                    buy_step_log = info['step']
                else:
                    sells.append(info['step'])
                    buy_step_log = None

    # Switch back to training mode
    policy.train()

     # 3 Performance Metrics
    final  = portfolio[-1]

    # Strategy P&L
    pnl    = (final - initial_cash) / initial_cash * 100

    # Buy & Hold benchmark
    bh_pnl = (prices[-1] / prices[0] - 1) * 100

    # Average holding duration
    avg_hold = np.mean(hold_durations) if hold_durations else 0

    print(f"\n  Policy NN  : ${final:>10,.2f}  ({pnl:+.2f}%)")
    print(f"  Buy & Hold : ${initial_cash*prices[-1]/prices[0]:>10,.2f}  ({bh_pnl:+.2f}%)")
    print(f"  Alpha      : {pnl - bh_pnl:+.2f}%")
    print(f"  Trades     : {len(buys)} BUY + {len(sells)} SELL")
    print(f"  Avg hold   : {avg_hold:.0f} steps per trade")
    return portfolio, prices, buys, sells


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    """
    Main execution block.

    This ensures the code runs ONLY when this file is executed directly:
        python train.py

    It will NOT run if this file is imported as a module.
    """
     
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Train the RL Agent
    # ==========================================================
    # Returns:
    #   policy            → Actor network π(a|s)
    #   value             → Critic network V(s)
    #   trainer           → Training object (optimizer, configs)
    #   rewards_log       → Episode reward history
    #   pnl_log           → Episode P&L history
    #   action_dist_log   → Action usage statistics
    #
    # num_episodes=300 means:
    #   300 full trading simulations

    policy, value, trainer, rewards_log, pnl_log, action_dist_log = train(
        num_episodes=300, initial_cash=10_000)

    # Save Model Checkpoint
    # ==========================================================
    # Saves:
    #   - Policy network weights
    #   - Value network weights
    #
    # Allows:
    #   - Reloading model later
    #   - Continuing training
    #   - Deployment
    #
    # state_dict() contains only learned parameters (not architecture).

    ##torch.save({'policy': policy.state_dict(), 'value': value.state_dict()},
              ## os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ##  'reinforce_checkpoint.pt'))
   ## print("  Checkpoint saved → reinforce_checkpoint.pt")

    #  Deterministic Evaluation
    # ==========================================================
    # Runs trained model using:
    #   action = argmax π(a|s)
    #
    # Measures:
    #   - Final portfolio value
    #   - Alpha vs Buy & Hold
    #   - Trade statistics
    #
    # This simulates real-world deployment behavior.

    eval_port, eval_prices, buys, sells = evaluate(policy, initial_cash=10_000)

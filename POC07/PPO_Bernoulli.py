"""
=============================================================================
PPO + Bernoulli Policy — Fraud Detection
=============================================================================
Architecture
────────────
  State   : 5 features per transaction (amount, time, velocity, geo, device)
  Action  : {0: Allow, 1: Block} — Bernoulli(σ(π_θ(s)))
  Policy  : 2-layer MLP actor  → sigmoid → Bernoulli distribution
  Value   : 2-layer MLP critic → V(s) baseline
  Algorithm: PPO with clipped surrogate + entropy bonus + clipped VF loss
  GAE     : Generalised Advantage Estimation (γ=0.99, λ=0.95)
  Optim   : Adam (separate LR for actor & critic)
=============================================================================
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────
# Detect GPU if available, else use CPU.
# Training on GPU is significantly faster for larger networks / batch sizes.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## print(f"  Using device: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    """
    Central configuration for PPO training.

    Keeping everything here makes experiments easy.
    Changing learning rate / clip / entropy is now one-line change.
    """

    # ── Environment settings ──────────────────────────────────────────────────
    # state_dim  : number of input features describing each transaction
    # fraud_rate : fraction of transactions that are genuinely fraudulent
    # batch_size : how many transactions to sample per rollout episode
    state_dim: int = 5
    fraud_rate: float = 0.25
    batch_size: int = 512  # transitions per rollout

    # ── Discount & GAE ────────────────────────────────────────────────────────
    # gamma : future reward discount factor (closer to 1 → longer-term thinking)
    # lam   : GAE λ — trades off bias vs variance in advantage estimates
    #         lam=1 → full MC returns (high variance), lam=0 → pure TD (high bias)
    gamma: float = 0.99
    lam: float = 0.95

    # ── PPO clipping ──────────────────────────────────────────────────────────
    # clip_eps    : max allowed policy ratio deviation per update step
    #               keeps updates conservative so training stays stable
    # vf_clip_eps : same clipping idea applied to the value function loss
    clip_eps: float = 0.20
    vf_clip_eps: float = 0.20  # value function clip

    # ── Optimisation ──────────────────────────────────────────────────────────
    # lr_actor  : actor network learning rate (smaller than critic is common)
    # lr_critic : critic network learning rate (needs faster convergence)
    # lr_decay  : multiplicative LR decay applied each episode via ExponentialLR
    # epochs    : number of gradient update passes over each collected rollout
    # mini_batch: size of each mini-batch drawn from the rollout buffer
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_decay: float = 0.999  # per-episode ExponentialLR decay
    epochs: int = 5  # PPO update epochs per rollout
    mini_batch: int = 128

    # ── Loss coefficients ─────────────────────────────────────────────────────
    # entropy_coef : weight of the entropy bonus in the total loss
    #                encourages exploration; prevents policy from collapsing
    #                to a deterministic action too early
    # vf_coef      : weight of the critic (value function) loss
    entropy_coef: float = 0.03  # fixed — avoids collapse
    vf_coef: float = 0.5

    # ── Regularisation ────────────────────────────────────────────────────────
    # max_grad_norm : gradient norm clipping threshold; prevents exploding gradients
    # weight_decay  : L2 regularisation penalty in Adam; reduces overfitting
    max_grad_norm: float = 0.5  # global gradient clipping
    weight_decay: float = 1e-4  # Adam L2

    # ── Network architecture ──────────────────────────────────────────────────
    # hidden : number of neurons in each hidden layer of actor and critic
    hidden: int = 64

    # ── Training schedule ─────────────────────────────────────────────────────
    # total_episodes : total number of collect→update cycles to run
    # log_every      : print a summary row every N episodes
    # eval_n         : number of held-out transactions for threshold tuning
    # seed           : random seed for reproducibility
    total_episodes: int = 400
    log_every: int = 20
    eval_n: int = 3000
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Transaction:
    # Raw transaction features — stored before normalisation
    amount: float
    time_of_day: float  # [0, 1]
    velocity: float  # [0, 1]
    geo_risk: float  # [0, 1]
    device_trust: float  # [0, 1]
    is_fraud: bool

    @property
    def state(self) -> np.ndarray:
        """
        Normalise raw features to approximately [-1, 1] for stable training.

        Why normalise?
        - Neural networks converge faster when inputs have similar scale.
        - Tanh activations are most sensitive near 0, so centring inputs helps.

        Transformations applied:
        - amount      : log-compressed then linearly mapped to [-1, 1]
                        log1p handles the heavy right tail of transaction amounts
        - time_of_day : already in [0,1], shifted to [-1, 1]
        - velocity    : already in [0,1], shifted to [-1, 1]
        - geo_risk    : already in [0,1], shifted to [-1, 1]
        - device_trust: already in [0,1], shifted to [-1, 1]
        """
        return np.array(
            [
                np.log1p(self.amount) / np.log1p(10_000) * 2 - 1,  # log-scale amount
                self.time_of_day * 2 - 1,
                self.velocity * 2 - 1,
                self.geo_risk * 2 - 1,
                self.device_trust * 2 - 1,
            ],
            dtype=np.float32,
        )

class FraudEnvironment:
    """
    Synthetic fraud detection environment.

    This simulates real-world fraud patterns:
    - Fraud transactions statistically differ from legit ones.
    - Reward is asymmetric (missing fraud is more costly).

    This makes it a cost-sensitive classification problem
    framed as a Reinforcement Learning task.

    Reward shaping
    ──────────────
    TP  +2.0   Blocked genuine fraud
    TN  +0.5   Allowed legitimate transaction
    FN  -2.0   Missed fraud  (most costly)
    FP  -0.8   False alarm   (customer friction)

    Fraud transactions are statistically distinct:
      - Higher amounts (lognormal with larger μ)
      - Odd-hour activity (0–6 AM, 70% of the time)
      - High velocity (Beta(5,2))
      - High geo risk (Beta(4,2))
      - Low device trust (Beta(2,5))
    """

    # Reward lookup table: key = (is_fraud, action)
    # This asymmetric design penalises missed fraud (FN) more than false alarms (FP)
    # because the real-world cost of undetected fraud is typically much higher.
    REWARD: Dict[Tuple[bool, int], float] = {
        (True, 1): +2.0,   # TP — correctly blocked fraud
        (False, 0): +0.5,  # TN — correctly allowed legit transaction
        (True, 0): -2.0,   # FN — fraud slipped through; most costly mistake
        (False, 1): -0.8,  # FP — legitimate transaction wrongly blocked
    }

    def __init__(self, cfg: PPOConfig, seed: int = 42):
        self.cfg = cfg
        # Use numpy's modern Generator for reproducible, seed-able sampling
        self.rng = np.random.default_rng(seed)

    def _sample(self) -> Transaction:
        """
        Generate a single synthetic transaction.

        Fraud vs. legit distributions are intentionally separable
        but overlapping — mimicking real financial data where
        not all signals perfectly distinguish the two classes.
        """
        is_fraud = self.rng.random() < self.cfg.fraud_rate

        if is_fraud:
            # Fraud profile:
            # - Larger amounts (lognormal μ=7.5 vs 5.0 for legit)
            # - 65% chance of occurring between midnight and 6 AM
            # - High transaction velocity (many transactions in short window)
            # - High geographic risk (unusual country / location)
            # - Low device trust (unrecognised device or emulator)
            return Transaction(
                amount=float(self.rng.lognormal(7.5, 1.2)),
                time_of_day=(
                    float(self.rng.uniform(0, 6))
                    if self.rng.random() < 0.65
                    else float(self.rng.uniform(0, 24))
                ) / 24,
                velocity=float(self.rng.beta(5, 2)),
                geo_risk=float(self.rng.beta(4, 2)),
                device_trust=float(self.rng.beta(2, 5)),
                is_fraud=True,
            )

        # Legit profile:
        # - Smaller amounts (lognormal μ=5.0)
        # - Activity concentrated in business hours (8 AM – 8 PM)
        # - Low velocity, low geo risk, high device trust
        return Transaction(
            amount=float(self.rng.lognormal(5.0, 0.8)),
            time_of_day=float(self.rng.uniform(8, 20)) / 24,
            velocity=float(self.rng.beta(2, 6)),
            geo_risk=float(self.rng.beta(2, 5)),
            device_trust=float(self.rng.beta(5, 2)),
            is_fraud=False,
        )

    def batch(self, n: int) -> List[Transaction]:
        """Sample a list of n independent transactions."""
        return [self._sample() for _ in range(n)]

    def step(self, tx: Transaction, action: int) -> float:
        """
        Return the reward for taking `action` on transaction `tx`.
        Looks up the (is_fraud, action) pair in the REWARD table.
        """
        return self.REWARD[(tx.is_fraud, action)]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  NEURAL NETWORKS
# ──────────────────────────────────────────────────────────────────────────────


def _init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """
    Orthogonal initialisation — proven stable for policy gradients.

    Why orthogonal?
    - Preserves gradient norm across layers at initialisation.
    - Empirically reduces variance during early training vs. random init.
    - The `gain` scales the singular values:
        gain=√2 for Tanh layers (compensates for the derivative < 1),
        gain=1.0 for the value output layer (no activation squashing).
    - Biases are zeroed so initial outputs are centered.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class ActorNetwork(nn.Module):
    """
    Bernoulli policy: π_θ(a|s) = Bernoulli(σ(MLP(s)))
    Why Bernoulli?
    - Action space is binary: {Allow=0, Block=1}
    - We model P(Block | state)
    - Sigmoid converts logits to probability
    - Bernoulli distribution allows:
        log_prob()  → required for PPO ratio
        entropy()   → encourages exploration

    Output: scalar logit → sigmoid → probability of blocking (action=1).
    torch.distributions.Bernoulli gives us:
      - log_prob(a)  correctly for both a=0 and a=1
      - entropy()    H(p) = -p log p - (1-p) log(1-p)
    """

    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        # Two hidden Tanh layers followed by a single linear output neuron.
        # Tanh is preferred over ReLU for policy networks because:
        #   - Output is bounded in (-1, 1), preventing extreme pre-activations.
        #   - Smooth gradients help the policy learn stable probability distributions.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),  # scalar logit (no activation here)
        )

        # Apply orthogonal init with gain=√2 to all linear layers
        self.apply(lambda m: _init_weights(m, gain=np.sqrt(2)))

        # Override the output layer with a much smaller gain.
        # This keeps initial logits close to 0 → probabilities near 0.5,
        # so the policy starts close to random (maximum exploration).
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> Bernoulli:
        """
        Forward pass — returns a Bernoulli distribution object.

        Steps:
          1. Pass state through the MLP → scalar logit per sample.
          2. squeeze(-1) converts shape (N, 1) → (N,) for element-wise ops.
          3. Bernoulli(logits=logit) internally applies sigmoid:
               p = σ(logit) = P(action=1 | state)
        """
        logit = self.net(x).squeeze(-1)  # (N,)
        return Bernoulli(logits=logit)

    def get_action(self, x: torch.Tensor):
        """
        Sample an action stochastically from the policy distribution.

        Used during rollout collection — we want exploration, not greedy picks.
        Returns:
          action   : sampled binary action (0 or 1)
          log_prob : log π_θ(action | state) — stored for PPO importance ratio
          entropy  : H[π(·|s)] — used in entropy bonus
          dist     : the Bernoulli distribution object (for further inspection)
        """
        dist = self.forward(x)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), dist

    def evaluate(self, x: torch.Tensor, action: torch.Tensor):
        """
        Re-evaluate previously sampled actions under the current (updated) policy.

        This is the key operation in the PPO update step:
        - We collect actions under the OLD policy (π_old).
        - After updating θ, we need log π_new(a|s) to compute the ratio r = π_new/π_old.
        - We also fetch entropy for the bonus term.

        Returns:
          log_prob : log π_θ_new(action | state)
          entropy  : H[π_θ_new(·|s)]
        """
        dist = self.forward(x)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

class CriticNetwork(nn.Module):
    """
    Value function V(s) — scalar output, no activation.

    Purpose:
    - Estimates expected future reward from state s.
    - Used as the baseline in GAE to reduce variance in policy gradient estimates.
    - A good critic makes advantages more accurate, speeding up learning.

    Architecture mirrors the actor but outputs a single unbounded scalar V(s).
    No final activation because V(s) can be any real number.
    """

    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),  # scalar V(s)
        )

        # gain=1.0 for the critic — the value output has no activation,
        # so we don't need to compensate for activation shrinkage.
        self.apply(lambda m: _init_weights(m, gain=1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(-1): (N, 1) → (N,) so values align with reward tensors
        return self.net(x).squeeze(-1)  # (N,)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROLLOUT BUFFER
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RolloutBuffer:
    """
    Stores one batch of collected experience and pre-computes GAE advantages.

    All tensors have shape (batch_size,) unless noted.

    Fields
    ──────
    states     : normalised state vectors, shape (N, state_dim)
    actions    : sampled binary actions {0, 1}, shape (N,)
    rewards    : immediate rewards r_t, shape (N,)
    log_probs  : log π_old(a_t | s_t) — old policy; used in PPO ratio
    values     : V_old(s_t) — old critic estimates; used in GAE & VF clipping
    advantages : GAE-estimated advantages A_t (normalised)
    returns    : target values for critic regression = A_t + V_t (normalised)
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor  # log π_old(a|s)
    values: torch.Tensor  # V_old(s)
    advantages: torch.Tensor
    returns: torch.Tensor  # = advantages + values (for VF training)

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        lam: float,
        adv_clip: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation (GAE, Schulman et al. 2015).

        GAE interpolates between Monte-Carlo returns (low bias, high variance)
        and one-step TD errors (high bias, low variance) via λ ∈ [0, 1].

        Recursion (backwards through time):
          δₜ = rₜ + γ · V(sₜ₊₁) − V(sₜ)   ← TD error
          Aₜ = δₜ + γ · λ · Aₜ₊₁            ← discounted accumulation

        The final time step has no next state, so V(s_{T+1}) = 0.

        Post-processing:
          - Clip advantages to [-adv_clip, +adv_clip] to prevent outliers
            from causing large gradient steps.
          - Z-score normalise advantages within the batch so their scale
            doesn't depend on the reward magnitude.
          - Z-score normalise returns so the critic regression target has
            unit variance — stabilises VF training.

        Returns:
          adv : normalised advantage estimates, shape (N,)
          ret : normalised return targets for critic, shape (N,)
        """
        n = len(rewards)
        adv = torch.zeros(n, dtype=torch.float32)
        gae = 0.0

        # Iterate backwards so each Aₜ can reference Aₜ₊₁ already computed
        for t in reversed(range(n)):
            # Bootstrap V(s_{t+1}) — zero at episode boundary (no next state)
            next_val = values[t + 1].item() if t + 1 < n else 0.0
            delta = rewards[t].item() + gamma * next_val - values[t].item()
            gae = delta + gamma * lam * gae
            adv[t] = gae

        # Returns = advantages + baseline values (used as critic regression target)
        ret = adv + values

        # Clip then normalise advantages to reduce the impact of rare large rewards
        adv = adv.clamp(-adv_clip, adv_clip)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Per-batch z-score returns for stable VF regression
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        return adv, ret

    @classmethod
    def from_rollout(
        cls,
        txs: List[Transaction],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        cfg: PPOConfig,
    ):
        """
        Construct a RolloutBuffer from raw rollout data.

        Steps:
          1. Convert list of Transaction objects to a stacked state tensor.
          2. Run GAE to get advantages and return targets.
          3. Detach log_probs and values so they're treated as fixed
             constants during the PPO update (they belong to the OLD policy).
        """
        states = torch.tensor(
            np.stack([tx.state for tx in txs]),
            dtype=torch.float32,
            device=DEVICE,
        )

        adv, ret = cls.compute_gae(rewards, values, cfg.gamma, cfg.lam)

        return cls(
            states=states,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs.detach(),  # stop grad — these are π_old
            values=values.detach(),  # stop grad — these are V_old
            advantages=adv.to(DEVICE),
            returns=ret.to(DEVICE),
        )

# ──────────────────────────────────────────────────────────────────────────────
# 5.  PPO AGENT
# ──────────────────────────────────────────────────────────────────────────────


class PPOAgent:

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

        # Seed both PyTorch and NumPy for reproducible weight init and sampling
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Instantiate networks and move them to the target device
        self.actor = ActorNetwork(cfg.state_dim, cfg.hidden).to(DEVICE)
        self.critic = CriticNetwork(cfg.state_dim, cfg.hidden).to(DEVICE)

        # Separate optimisers allow independent learning rates for actor & critic.
        # The critic typically benefits from a higher LR because it's a simpler
        # regression task (predict scalar V) vs. the actor's policy optimisation.
        self.actor_optim = Adam(
            self.actor.parameters(),
            lr=cfg.lr_actor,
            weight_decay=cfg.weight_decay,
        )
        self.critic_optim = Adam(
            self.critic.parameters(),
            lr=cfg.lr_critic,
            weight_decay=cfg.weight_decay,
        )

        # Learning rate schedulers decay the LR multiplicatively each episode.
        # This provides a large initial step size for fast early learning,
        # then gradually shrinks updates for stable fine-tuning near convergence.
        self.actor_sched = ExponentialLR(self.actor_optim, gamma=cfg.lr_decay)
        self.critic_sched = ExponentialLR(self.critic_optim, gamma=cfg.lr_decay)

    # ── ROLLOUT ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect(
        self,
        env: FraudEnvironment,
    ) -> Tuple[RolloutBuffer, List[Transaction], np.ndarray, np.ndarray]:
        """
        Collect one rollout batch by running the current policy in the environment.

        @torch.no_grad() is critical here — during rollout we only need forward
        passes for inference; no gradient computation is needed or wanted.
        This saves memory and speeds up data collection.

        Steps:
          1. Sample a batch of transactions from the environment.
          2. Stack their normalised states into a tensor.
          3. Actor samples actions stochastically (exploration).
          4. Critic estimates V(s) for each state.
          5. Environment returns a reward for each (transaction, action) pair.
          6. Pack everything into a RolloutBuffer (which computes GAE).

        Returns:
          buf     : RolloutBuffer with pre-computed advantages/returns
          txs     : raw Transaction objects (for metric logging)
          actions : numpy array of sampled actions
          rewards : numpy array of received rewards
        """
        txs = env.batch(self.cfg.batch_size)

        states = torch.tensor(
            np.stack([tx.state for tx in txs]),
            dtype=torch.float32,
            device=DEVICE,
        )

        actions, log_probs, _, _ = self.actor.get_action(states)
        values = self.critic(states)

        rewards = torch.tensor(
            [
                env.step(tx, int(a))
                for tx, a in zip(txs, actions.cpu().numpy())
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        buf = RolloutBuffer.from_rollout(
            txs,
            actions,
            rewards,
            log_probs,
            values,
            self.cfg,
        )

        return (
            buf,
            txs,
            actions.cpu().numpy(),
            rewards.cpu().numpy(),
        )

    # ── PPO UPDATE ────────────────────────────────────────────────────────────

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        """
        Run the PPO clipped surrogate update over the collected rollout buffer.

        PPO loss components (computed per mini-batch):
        ────────────────────────────────────────────
        L_clip  = E[ min( r·A,  clip(r, 1±ε)·A ) ]
          - r = π_new(a|s) / π_old(a|s)  — importance sampling ratio
          - Clipping prevents overly large policy updates.
          - Taking the min acts as a pessimistic bound.

        L_vf    = E[ max( (V_new − ret)²,  (V_clip − ret)² ) ]
          - Clipped value loss mirrors actor clipping to bound critic changes.
          - V_clip restricts how far V_new can move from V_old per step.

        L_ent   = H[ π(·|s) ]
          - Entropy bonus discourages premature policy collapse to a single action.
          - Especially important early in training when the critic is noisy.

        Combined loss (minimised):
          L_total = −L_clip  +  c_vf · L_vf  −  c_ent · L_ent

        The rollout buffer is shuffled and split into mini-batches each epoch
        to reduce correlation between gradient updates and improve data efficiency.

        Returns a dictionary of mean loss/metric values across all mini-batches.
        """
        actor_losses = []
        critic_losses = []
        entropies = []
        kls = []

        for _ in range(self.cfg.epochs):

            # Shuffle indices each epoch so mini-batches see different orderings
            idx = torch.randperm(len(buf.states), device=DEVICE)

            for start in range(0, len(buf.states), self.cfg.mini_batch):
                mb = idx[start:start + self.cfg.mini_batch]

                # Slice the mini-batch from the rollout buffer
                s = buf.states[mb]
                a = buf.actions[mb]
                adv = buf.advantages[mb]
                ret = buf.returns[mb]
                olp = buf.log_probs[mb]  # old log-probabilities (fixed)
                ov = buf.values[mb]      # old value estimates (fixed)

                # ── Actor loss ─────────────────────────────────────────────────
                # Re-evaluate actions under the CURRENT (updated) policy
                new_lp, ent = self.actor.evaluate(s, a)

                # Importance sampling ratio: r = exp(log π_new − log π_old)
                # We work in log-space for numerical stability.
                ratio = (new_lp - olp).exp()

                # Clipped surrogate objective
                # - Without clipping: maximising r·A can cause huge policy shifts.
                # - Clipping to [1−ε, 1+ε] keeps updates within a trust region.
                clip_ratio = ratio.clamp(
                    1 - self.cfg.clip_eps,
                    1 + self.cfg.clip_eps,
                )

                actor_loss = -torch.min(
                    ratio * adv,
                    clip_ratio * adv,
                ).mean()

                entropy = ent.mean()

                # ── Critic loss (clipped VF loss) ──────────────────────────────
                new_v = self.critic(s)

                # Normalise old values to the same z-score scale as returns.
                # Without this, ov and ret would be on incompatible scales
                # because returns were already z-scored in compute_gae.
                ov_norm = (
                    (ov - buf.values.mean()) /
                    (buf.values.std() + 1e-8)
                )

                # V_clip: restrict how much V_new can deviate from V_old per step
                v_clip = ov_norm + (
                    new_v - ov_norm
                ).clamp(
                    -self.cfg.vf_clip_eps,
                    self.cfg.vf_clip_eps,
                )

                # Take the max of both MSE losses (pessimistic bound, same spirit as actor)
                critic_loss = torch.max(
                    (new_v - ret).pow(2),
                    (v_clip - ret).pow(2),
                ).mean()

                # ── Combined loss ──────────────────────────────────────────────
                # Negative entropy because we MAXIMISE entropy (subtract from minimised loss)
                loss = (
                    actor_loss
                    + self.cfg.vf_coef * critic_loss
                    - self.cfg.entropy_coef * entropy
                )

                # ── Backprop ───────────────────────────────────────────────────
                # Zero both optimiser gradients before backward to avoid accumulation
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()

                # Global gradient norm clipping prevents destabilising large updates
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.cfg.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.cfg.max_grad_norm,
                )

                self.actor_optim.step()
                self.critic_optim.step()

                # ── Logging ────────────────────────────────────────────────────
                with torch.no_grad():
                    # Approximate KL divergence: KL(π_old || π_new) ≈ E[log π_old − log π_new]
                    # Used as a diagnostic — large KL signals the policy changed too much.
                    kl = (olp - new_lp).mean().item()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                kls.append(kl)

        # Step LR schedulers once per episode (not per mini-batch)
        self.actor_sched.step()
        self.critic_sched.step()

        # Return mean statistics across all mini-batches and epochs for logging
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "mean_kl": float(np.mean(kls)),
            "lr_actor": self.actor_optim.param_groups[0]["lr"],
            "lr_critic": self.critic_optim.param_groups[0]["lr"],
        }

    # ── INFERENCE ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        state: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[int, float]:
        """
        Deterministic inference for a single transaction state.

        Instead of sampling from the Bernoulli distribution (stochastic),
        we threshold the block probability for deterministic deployment:
          action = 1  if P(block | state) >= threshold  else 0

        The threshold can be tuned post-training to balance precision/recall.

        Returns:
          action : 0 (allow) or 1 (block)
          prob   : raw probability of blocking P(block | state)
        """
        x = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        dist = self.actor(x)
        prob = dist.probs.item()  # σ(logit) — probability of blocking

        return int(prob >= threshold), float(prob)

    @torch.no_grad()
    def predict_batch(
        self,
        states: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised batch inference — efficient for evaluation and threshold search.

        Returns:
          actions : binary numpy array of predictions (N,)
          probs   : block probabilities for each sample (N,)
        """
        x = torch.tensor(
            states,
            dtype=torch.float32,
            device=DEVICE,
        )

        dist = self.actor(x)
        probs = dist.probs.cpu().numpy()

        return (probs >= threshold).astype(int), probs

# ──────────────────────────────────────────────────────────────────────────────
# 6.  THRESHOLD TUNING  (maximise F1 on held-out set)
# ──────────────────────────────────────────────────────────────────────────────


def tune_threshold(
    agent: PPOAgent,
    env: FraudEnvironment,
    n: int = 3000,
) -> Tuple[float, float]:
    """
    Search for the classification threshold that maximises F1 score on a
    fresh held-out set of transactions (not seen during training).

    Why tune the threshold?
    - The actor outputs P(block | state) which is a continuous score.
    - The default threshold of 0.5 is rarely optimal for imbalanced classes.
    - With fraud_rate=0.25 and asymmetric rewards, the optimal threshold
      is often lower than 0.5 to catch more fraud at the cost of more false alarms.

    Approach:
    - Generate n held-out transactions with a different random seed.
    - Sweep thresholds from 0.05 to 0.95 in 90 steps.
    - Pick the threshold with the highest F1 = 2·P·R / (P+R).

    F1 is chosen as the target metric because it balances precision and recall,
    which is appropriate for the imbalanced fraud detection setting.

    Returns:
      best_thr : threshold value in [0.05, 0.95]
      best_f1  : F1 score achieved at that threshold
    """
    txs = env.batch(n)
    states = np.stack([tx.state for tx in txs])
    labels = np.array([int(tx.is_fraud) for tx in txs])

    _, probs = agent.predict_batch(states)

    best_f1 = 0.0
    best_thr = 0.5

    for thr in np.linspace(0.05, 0.95, 90):
        preds = (probs >= thr).astype(int)

        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        # Precision: of all blocked transactions, how many were truly fraud?
        pr = tp / (tp + fp + 1e-8)

        # Recall: of all fraud transactions, how many did we catch?
        rc = tp / (tp + fn + 1e-8)

        # F1: harmonic mean of precision and recall
        f1 = 2 * pr * rc / (pr + rc + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    print(
        f"\n  Threshold tuning → best threshold = {best_thr:.2f}  |  "
        f"F1 = {best_f1:.4f}"
    )

    return best_thr, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# 7.  METRICS TRACKER
# ──────────────────────────────────────────────────────────────────────────────


class MetricsTracker:
    """
    Accumulates per-episode training statistics and confusion matrix counts.

    Metrics are stored as a running history list of dicts, one per episode.
    Cumulative TP/TN/FP/FN counts grow across all episodes (not per-episode),
    giving a stable view of overall model performance over the full training run.

    Use smooth() to apply a moving-average filter before plotting,
    which reduces the noise from episode-to-episode variability.
    """

    def __init__(self):
        self.history: List[dict] = []

        # Cumulative confusion matrix counts across all training episodes
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0

    def update(
        self,
        txs: List[Transaction],
        actions: np.ndarray,
        rewards: np.ndarray,
        losses: Dict[str, float],
        episode: int,
    ):
        """
        Process one episode's results and append a metrics snapshot.

        Confusion matrix is updated cumulatively — each episode adds to the totals
        from all previous episodes. This means metrics improve in signal-to-noise
        ratio over time since they're averaged over ever more samples.
        """
        # Update running confusion matrix with this episode's predictions
        for tx, a in zip(txs, actions):
            if tx.is_fraud and a == 1:
                self._tp += 1   # True Positive
            elif not tx.is_fraud and a == 0:
                self._tn += 1   # True Negative
            elif tx.is_fraud and a == 0:
                self._fn += 1   # False Negative (missed fraud)
            else:
                self._fp += 1   # False Positive (false alarm)

        # Compute derived metrics from cumulative confusion matrix
        prec = self._tp / (self._tp + self._fp + 1e-8)  # precision
        rec = self._tp / (self._tp + self._fn + 1e-8)   # recall
        f1 = 2 * prec * rec / (prec + rec + 1e-8)       # F1 score

        total = self._tp + self._tn + self._fp + self._fn
        acc = (self._tp + self._tn) / max(total, 1)

        # Append a full snapshot of this episode's metrics to history
        self.history.append(
            {
                "episode": episode,
                "avg_reward": float(rewards.mean()),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
                "actor_loss": losses["actor_loss"],
                "critic_loss": losses["critic_loss"],
                "entropy": losses["entropy"],
                "mean_kl": losses["mean_kl"],
                "lr_actor": losses["lr_actor"],
                # Confusion matrix snapshot (cumulative totals)
                "tp": self._tp,
                "tn": self._tn,
                "fp": self._fp,
                "fn": self._fn,
            }
        )

    @property
    def latest(self) -> dict:
        """Return the most recent episode's metrics snapshot."""
        return self.history[-1] if self.history else {}

    def smooth(self, key: str, n: int = 15) -> Tuple[np.ndarray, List[int]]:
        """
        Apply a uniform moving-average filter of window size n to a metric series.

        The 'valid' convolution mode trims n-1 leading samples where the window
        isn't fully populated, so the returned episode indices are offset by n-1.

        Returns:
          sm  : smoothed values, length = (total_episodes - n + 1)
          eps : corresponding episode indices for x-axis alignment
        """
        vals = np.array([h[key] for h in self.history], dtype=float)
        sm = np.convolve(vals, np.ones(n) / n, mode="valid")

        return sm, list(range(n - 1, len(vals)))

# ──────────────────────────────────────────────────────────────────────────────
# 8.  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: PPOConfig) -> Tuple[PPOAgent, MetricsTracker, float]:
    """
    Main training loop: iterates collect → update → log for total_episodes.

    Each episode:
      1. collect() : run the current policy to gather a batch of transitions.
      2. update()  : run PPO mini-batch updates over the collected buffer.
      3. tracker.update() : record metrics for monitoring.

    After training, tune_threshold() is called on a fresh held-out set
    to find the best classification threshold (maximises F1).

    Returns:
      agent     : trained PPOAgent (actor + critic weights)
      tracker   : full training history
      threshold : optimal classification threshold found by tuning
    """
    env = FraudEnvironment(cfg)
    agent = PPOAgent(cfg)
    tracker = MetricsTracker()

    # Print model summary before training starts
    print("=" * 82)
    print("  PPO + BERNOULLI POLICY  ·  FRAUD DETECTION  ")
    print(
        f"  Actor: {sum(p.numel() for p in agent.actor.parameters())} params  |  "
        f"Critic: {sum(p.numel() for p in agent.critic.parameters())} params  |  "
        f"Device: {DEVICE}"
    )
    print("=" * 82)
    print(
        f"{'Ep':>6} | {'AvgRew':>7} | {'F1':>6} | {'Acc':>6} | "
        f"{'Prec':>6} | {'Rec':>6} | {'KL':>7} | {'Entropy':>8} | {'ActorL':>8}"
    )
    print("-" * 82)

    for ep in range(1, cfg.total_episodes + 1):
        # Collect: sample transactions and run the policy to gather experience
        buf, txs, actions, rewards = agent.collect(env)

        # Update: run PPO gradient steps over the collected buffer
        losses = agent.update(buf)

        # Track: record metrics for this episode
        tracker.update(txs, actions, rewards, losses, ep)

        # Periodically print a summary row to the console
        if ep % cfg.log_every == 0:
            m = tracker.latest
            print(
                f"{ep:>6} | {m['avg_reward']:>7.3f} | {m['f1']:>6.3f} | "
                f"{m['accuracy']:>6.3f} | {m['precision']:>6.3f} | "
                f"{m['recall']:>6.3f} | {m['mean_kl']:>7.4f} | "
                f"{m['entropy']:>8.4f} | {m['actor_loss']:>8.4f}"
            )

    # Print final cumulative metrics after all episodes complete
    print("=" * 82)
    m = tracker.latest
    print(f"\n  ── Final Results (episode {cfg.total_episodes}) ──")
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  F1 Score   : {m['f1']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print("\n  Confusion Matrix (cumulative)")
    print(f"  TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")

    # Tune threshold on a fresh environment (different seed → unseen transactions)
    threshold, _ = tune_threshold(
        agent,
        FraudEnvironment(cfg, seed=999),
        cfg.eval_n,
    )

    return agent, tracker, threshold


# ──────────────────────────────────────────────────────────────────────────────
# 10.  INFERENCE DEMO
# ──────────────────────────────────────────────────────────────────────────────

def demo_inference(agent: PPOAgent, threshold: float, n: int = 15):
    """
    Run a small qualitative demo showing the trained agent's predictions.

    Prints a table with the raw transaction features, the agent's block
    probability, the chosen action, the received reward, and whether the
    decision was correct (✓) or wrong (✗).

    Uses a fixed seed (1234) so the demo output is reproducible across runs.
    The same seed is used for both transaction generation and reward evaluation.
    """
    env = FraudEnvironment(PPOConfig(), seed=1234)
    txs = env.batch(n)

    print("\n" + "=" * 82)
    print(f"  INFERENCE DEMO  (threshold = {threshold:.2f})")
    print("=" * 82)
    print(
        f"{'Amount':>10} | {'Time':>5} | {'Vel':>5} | {'Geo':>5} | {'Dev':>5} | "
        f"{'Ground':>7} | {'P(blk)':>7} | {'Action':>6} | {'Reward':>7} | {'OK?':>4}"
    )
    print("-" * 82)

    correct = 0
    rewards_total = 0.0

    # Use the same seed for the evaluation environment to ensure consistent rewards
    env_eval = FraudEnvironment(PPOConfig(), seed=1234)

    for tx in txs:
        # Deterministic prediction using the tuned threshold
        action, prob = agent.predict(tx.state, threshold)
        reward = env_eval.step(tx, action)
        ok = tx.is_fraud == (action == 1)  # True if decision matches ground truth

        correct += ok
        rewards_total += reward

        print(
            f"${tx.amount:>8,.0f} | {tx.time_of_day:>5.2f} | "
            f"{tx.velocity:>5.2f} | {tx.geo_risk:>5.2f} | "
            f"{tx.device_trust:>5.2f} | "
            f"{'FRAUD' if tx.is_fraud else 'LEGIT':>7} | "
            f"{prob:>7.3f} | "
            f"{'BLOCK' if action == 1 else 'ALLOW':>6} | "
            f"{reward:>+7.1f} | "
            f"{'✓' if ok else '✗':>4}"
        )

    # Summary line with overall accuracy and cumulative reward for the demo batch
    print("-" * 82)
    print(
        f"  Accuracy: {correct}/{n}  ({100 * correct / n:.0f}%)  |  "
        f"Total reward: {rewards_total:+.1f}"
    )
    print("=" * 82)


# ──────────────────────────────────────────────────────────────────────────────
# 11.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Instantiate config — override specific values here for quick experiments.
    # All other hyperparameters default to the values defined in PPOConfig.
    cfg = PPOConfig(
        total_episodes=400,
        batch_size=512,
        hidden=64,
        lr_actor=3e-4,
        lr_critic=1e-3,
        entropy_coef=0.03,
        seed=42,
    )

    # Run full training loop; returns the trained agent, history, and best threshold
    agent, tracker, threshold = train(cfg)

    # Print a qualitative demo of the agent on 15 fresh transactions
    demo_inference(agent, threshold)
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## print(f"  Using device: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Environment
    state_dim: int         = 5
    fraud_rate: float      = 0.25
    batch_size: int        = 512       # transitions per rollout

    # Discount & GAE
    gamma: float           = 0.99
    lam: float             = 0.95

    # PPO clipping
    clip_eps: float        = 0.20
    vf_clip_eps: float     = 0.20      # value function clip

    # Optimisation
    lr_actor: float        = 3e-4
    lr_critic: float       = 1e-3
    lr_decay: float        = 0.999     # per-episode ExponentialLR decay
    epochs: int            = 5         # PPO update epochs per rollout
    mini_batch: int        = 128

    # Loss coefficients
    entropy_coef: float    = 0.03      # fixed — avoids collapse
    vf_coef: float         = 0.5

    # Regularisation
    max_grad_norm: float   = 0.5       # global gradient clipping
    weight_decay: float    = 1e-4      # Adam L2

    # Network
    hidden: int            = 64

    # Training
    total_episodes: int    = 400
    log_every: int         = 20
    eval_n: int            = 3000
    seed: int              = 42


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Transaction:
    amount: float
    time_of_day: float      # [0, 1]
    velocity: float         # [0, 1]
    geo_risk: float         # [0, 1]
    device_trust: float     # [0, 1]
    is_fraud: bool

    @property
    def state(self) -> np.ndarray:
        """Normalise to ~[-1, 1] for stable training."""
        return np.array([
            np.log1p(self.amount) / np.log1p(10_000) * 2 - 1,  # log-scale amount
            self.time_of_day * 2 - 1,
            self.velocity * 2 - 1,
            self.geo_risk * 2 - 1,
            self.device_trust * 2 - 1,
        ], dtype=np.float32)


class FraudEnvironment:
    """
    Synthetic fraud detection MDP.

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

    REWARD: Dict[Tuple[bool, int], float] = {
        (True,  1): +2.0,   # TP
        (False, 0): +0.5,   # TN
        (True,  0): -2.0,   # FN
        (False, 1): -0.8,   # FP
    }

    def __init__(self, cfg: PPOConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _sample(self) -> Transaction:
        is_fraud = self.rng.random() < self.cfg.fraud_rate
        if is_fraud:
            return Transaction(
                amount       = float(self.rng.lognormal(7.5, 1.2)),
                time_of_day  = (float(self.rng.uniform(0, 6)) if self.rng.random() < 0.65
                                else float(self.rng.uniform(0, 24))) / 24,
                velocity     = float(self.rng.beta(5, 2)),
                geo_risk     = float(self.rng.beta(4, 2)),
                device_trust = float(self.rng.beta(2, 5)),
                is_fraud     = True,
            )
        return Transaction(
            amount       = float(self.rng.lognormal(5.0, 0.8)),
            time_of_day  = float(self.rng.uniform(8, 20)) / 24,
            velocity     = float(self.rng.beta(2, 6)),
            geo_risk     = float(self.rng.beta(2, 5)),
            device_trust = float(self.rng.beta(5, 2)),
            is_fraud     = False,
        )

    def batch(self, n: int) -> List[Transaction]:
        return [self._sample() for _ in range(n)]

    def step(self, tx: Transaction, action: int) -> float:
        return self.REWARD[(tx.is_fraud, action)]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  NEURAL NETWORKS
# ──────────────────────────────────────────────────────────────────────────────

def _init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal initialisation — proven stable for policy gradients."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class ActorNetwork(nn.Module):
    """
    Bernoulli policy: π_θ(a|s) = Bernoulli(σ(MLP(s)))

    Output: scalar logit → sigmoid → probability of blocking (action=1).
    torch.distributions.Bernoulli gives us:
      - log_prob(a)  correctly for both a=0 and a=1
      - entropy()    H(p) = -p log p - (1-p) log(1-p)
    """

    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.apply(lambda m: _init_weights(m, gain=np.sqrt(2)))
        # Output layer: small init for stable early logits
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> Bernoulli:
        """Returns a Bernoulli distribution object."""
        logit = self.net(x).squeeze(-1)          # (N,)
        return Bernoulli(logits=logit)

    def get_action(self, x: torch.Tensor):
        """Sample action and return (action, log_prob, entropy, dist)."""
        dist   = self.forward(x)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), dist

    def evaluate(self, x: torch.Tensor, action: torch.Tensor):
        """Evaluate stored actions — used in PPO update."""
        dist     = self.forward(x)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Value function V(s) — scalar output, no activation."""

    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.apply(lambda m: _init_weights(m, gain=1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)            # (N,)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROLLOUT BUFFER
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """Stores one batch of experience and computes GAE advantages."""
    states:       torch.Tensor
    actions:      torch.Tensor
    rewards:      torch.Tensor
    log_probs:    torch.Tensor   # log π_old(a|s)
    values:       torch.Tensor   # V_old(s)
    advantages:   torch.Tensor
    returns:      torch.Tensor   # = advantages + values (for VF training)

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values:  torch.Tensor,
        gamma:   float,
        lam:     float,
        adv_clip: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation.
        δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)
        Aₜ = δₜ + γλ Aₜ₊₁
        """
        n   = len(rewards)
        adv = torch.zeros(n, dtype=torch.float32)
        gae = 0.0

        for t in reversed(range(n)):
            next_val = values[t + 1].item() if t + 1 < n else 0.0
            delta    = rewards[t].item() + gamma * next_val - values[t].item()
            gae      = delta + gamma * lam * gae
            adv[t]   = gae

        ret = adv + values

        # Clip then normalise advantages
        adv = adv.clamp(-adv_clip, adv_clip)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Per-batch z-score returns for stable VF regression
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        return adv, ret

    @classmethod
    def from_rollout(
        cls,
        txs:     List[Transaction],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        values:  torch.Tensor,
        cfg:     PPOConfig,
    ):
        states = torch.tensor(
            np.stack([tx.state for tx in txs]), dtype=torch.float32, device=DEVICE
        )
        adv, ret = cls.compute_gae(rewards, values, cfg.gamma, cfg.lam)
        return cls(
            states    = states,
            actions   = actions,
            rewards   = rewards,
            log_probs = log_probs.detach(),
            values    = values.detach(),
            advantages = adv.to(DEVICE),
            returns    = ret.to(DEVICE),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PPO AGENT
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent:

    def __init__(self, cfg: PPOConfig):
        self.cfg    = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.actor  = ActorNetwork(cfg.state_dim, cfg.hidden).to(DEVICE)
        self.critic = CriticNetwork(cfg.state_dim, cfg.hidden).to(DEVICE)

        self.actor_optim  = Adam(
            self.actor.parameters(),
            lr=cfg.lr_actor, weight_decay=cfg.weight_decay,
        )
        self.critic_optim = Adam(
            self.critic.parameters(),
            lr=cfg.lr_critic, weight_decay=cfg.weight_decay,
        )
        self.actor_sched  = ExponentialLR(self.actor_optim,  gamma=cfg.lr_decay)
        self.critic_sched = ExponentialLR(self.critic_optim, gamma=cfg.lr_decay)

    # ── ROLLOUT ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def collect(self, env: FraudEnvironment) -> Tuple[RolloutBuffer, List[Transaction]]:
        txs     = env.batch(self.cfg.batch_size)
        states  = torch.tensor(
            np.stack([tx.state for tx in txs]), dtype=torch.float32, device=DEVICE
        )
        actions, log_probs, _, _ = self.actor.get_action(states)
        values                   = self.critic(states)
        rewards = torch.tensor(
            [env.step(tx, int(a)) for tx, a in zip(txs, actions.cpu().numpy())],
            dtype=torch.float32, device=DEVICE,
        )
        buf = RolloutBuffer.from_rollout(txs, actions, rewards, log_probs, values, self.cfg)
        return buf, txs, actions.cpu().numpy(), rewards.cpu().numpy()

    # ── PPO UPDATE ────────────────────────────────────────────────────────────
    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        """
        PPO loss (per mini-batch):
          L_clip  = E[min(r·A,  clip(r, 1±ε)·A)]          actor
          L_vf    = E[max(v-ret)², (v_clip-ret)²]          critic
          L_ent   = H[π(·|s)]                              entropy bonus
          L_total = -L_clip + c_vf·L_vf - c_ent·L_ent
        """
        actor_losses, critic_losses, entropies, kls = [], [], [], []

        for _ in range(self.cfg.epochs):
            idx = torch.randperm(len(buf.states), device=DEVICE)

            for start in range(0, len(buf.states), self.cfg.mini_batch):
                mb = idx[start : start + self.cfg.mini_batch]

                s   = buf.states[mb]
                a   = buf.actions[mb]
                adv = buf.advantages[mb]
                ret = buf.returns[mb]
                olp = buf.log_probs[mb]
                ov  = buf.values[mb]

                # ── Actor ──────────────────────────────────────────────────
                new_lp, ent = self.actor.evaluate(s, a)
                ratio       = (new_lp - olp).exp()

                clip_ratio  = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                actor_loss  = -torch.min(ratio * adv, clip_ratio * adv).mean()
                entropy     = ent.mean()

                # ── Critic  (clipped VF loss) ──────────────────────────────
                new_v      = self.critic(s)
                # Normalise old values to same scale as returns (already z-scored)
                ov_norm    = (ov - buf.values.mean()) / (buf.values.std() + 1e-8)
                v_clip     = ov_norm + (new_v - ov_norm).clamp(
                    -self.cfg.vf_clip_eps, self.cfg.vf_clip_eps
                )
                critic_loss = torch.max(
                    (new_v - ret).pow(2),
                    (v_clip - ret).pow(2),
                ).mean()

                # ── Combined loss ──────────────────────────────────────────
                loss = (
                    actor_loss
                    + self.cfg.vf_coef * critic_loss
                    - self.cfg.entropy_coef * entropy
                )

                # ── Backprop ───────────────────────────────────────────────
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()

                # Global gradient norm clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(),  self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)

                self.actor_optim.step()
                self.critic_optim.step()

                # ── Logging ────────────────────────────────────────────────
                with torch.no_grad():
                    kl = (olp - new_lp).mean().item()   # approx KL

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                kls.append(kl)

        # Step LR schedulers once per episode
        self.actor_sched.step()
        self.critic_sched.step()

        return {
            "actor_loss":   float(np.mean(actor_losses)),
            "critic_loss":  float(np.mean(critic_losses)),
            "entropy":      float(np.mean(entropies)),
            "mean_kl":      float(np.mean(kls)),
            "lr_actor":     self.actor_optim.param_groups[0]["lr"],
            "lr_critic":    self.critic_optim.param_groups[0]["lr"],
        }

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, state: np.ndarray, threshold: float = 0.5) -> Tuple[int, float]:
        x    = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        dist = self.actor(x)
        prob = dist.probs.item()
        return int(prob >= threshold), float(prob)

    @torch.no_grad()
    def predict_batch(self, states: np.ndarray, threshold: float = 0.5):
        x    = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        dist = self.actor(x)
        probs = dist.probs.cpu().numpy()
        return (probs >= threshold).astype(int), probs


# ──────────────────────────────────────────────────────────────────────────────
# 6.  THRESHOLD TUNING  (maximise F1 on held-out set)
# ──────────────────────────────────────────────────────────────────────────────

def tune_threshold(agent: PPOAgent, env: FraudEnvironment, n: int = 3000) -> Tuple[float, float]:
    txs    = env.batch(n)
    states = np.stack([tx.state for tx in txs])
    labels = np.array([int(tx.is_fraud) for tx in txs])
    _, probs = agent.predict_batch(states)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 90):
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * rc / (pr + rc + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    print(f"\n  Threshold tuning → best threshold = {best_thr:.2f}  |  F1 = {best_f1:.4f}")
    return best_thr, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# 7.  METRICS TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:

    def __init__(self):
        self.history: List[dict] = []
        self._tp = self._tn = self._fp = self._fn = 0

    def update(
        self,
        txs:     List[Transaction],
        actions: np.ndarray,
        rewards: np.ndarray,
        losses:  Dict[str, float],
        episode: int,
    ):
        for tx, a in zip(txs, actions):
            if   tx.is_fraud  and a == 1: self._tp += 1
            elif not tx.is_fraud and a == 0: self._tn += 1
            elif tx.is_fraud  and a == 0: self._fn += 1
            else:                          self._fp += 1

        prec = self._tp / (self._tp + self._fp + 1e-8)
        rec  = self._tp / (self._tp + self._fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        acc  = (self._tp + self._tn) / max(self._tp + self._tn + self._fp + self._fn, 1)

        self.history.append({
            "episode":     episode,
            "avg_reward":  float(rewards.mean()),
            "precision":   prec,
            "recall":      rec,
            "f1":          f1,
            "accuracy":    acc,
            "actor_loss":  losses["actor_loss"],
            "critic_loss": losses["critic_loss"],
            "entropy":     losses["entropy"],
            "mean_kl":     losses["mean_kl"],
            "lr_actor":    losses["lr_actor"],
            "tp": self._tp, "tn": self._tn,
            "fp": self._fp, "fn": self._fn,
        })

    @property
    def latest(self) -> dict:
        return self.history[-1] if self.history else {}

    def smooth(self, key: str, n: int = 15) -> Tuple[np.ndarray, List[int]]:
        vals = np.array([h[key] for h in self.history], dtype=float)
        sm   = np.convolve(vals, np.ones(n) / n, mode="valid")
        return sm, list(range(n - 1, len(vals)))


# ──────────────────────────────────────────────────────────────────────────────
# 8.  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: PPOConfig) -> Tuple[PPOAgent, MetricsTracker, float]:
    env     = FraudEnvironment(cfg)
    agent   = PPOAgent(cfg)
    tracker = MetricsTracker()

    print("=" * 82)
    print("  PPO + BERNOULLI POLICY  ·  FRAUD DETECTION  ")
    print(f"  Actor: {sum(p.numel() for p in agent.actor.parameters())} params  |  "
          f"Critic: {sum(p.numel() for p in agent.critic.parameters())} params  |  "
          f"Device: {DEVICE}")
    print("=" * 82)
    print(f"{'Ep':>6} | {'AvgRew':>7} | {'F1':>6} | {'Acc':>6} | "
          f"{'Prec':>6} | {'Rec':>6} | {'KL':>7} | {'Entropy':>8} | {'ActorL':>8}")
    print("-" * 82)

    for ep in range(1, cfg.total_episodes + 1):
        buf, txs, actions, rewards = agent.collect(env)
        losses  = agent.update(buf)
        tracker.update(txs, actions, rewards, losses, ep)

        if ep % cfg.log_every == 0:
            m = tracker.latest
            print(f"{ep:>6} | {m['avg_reward']:>7.3f} | {m['f1']:>6.3f} | "
                  f"{m['accuracy']:>6.3f} | {m['precision']:>6.3f} | {m['recall']:>6.3f} | "
                  f"{m['mean_kl']:>7.4f} | {m['entropy']:>8.4f} | {m['actor_loss']:>8.4f}")

    print("=" * 82)
    m = tracker.latest
    print(f"\n  ── Final Results (episode {cfg.total_episodes}) ──")
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  F1 Score   : {m['f1']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"\n  Confusion Matrix (cumulative)")
    print(f"  TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")

    threshold, _ = tune_threshold(agent, FraudEnvironment(cfg, seed=999), cfg.eval_n)
    return agent, tracker, threshold





# ──────────────────────────────────────────────────────────────────────────────
# 10.  INFERENCE DEMO
# ──────────────────────────────────────────────────────────────────────────────

def demo_inference(agent: PPOAgent, threshold: float, n: int = 15):
    env = FraudEnvironment(PPOConfig(), seed=1234)
    txs = env.batch(n)

    print("\n" + "=" * 82)
    print(f"  INFERENCE DEMO  (threshold = {threshold:.2f})")
    print("=" * 82)
    print(f"{'Amount':>10} | {'Time':>5} | {'Vel':>5} | {'Geo':>5} | {'Dev':>5} | "
          f"{'Ground':>7} | {'P(blk)':>7} | {'Action':>6} | {'Reward':>7} | {'OK?':>4}")
    print("-" * 82)

    correct = 0
    rewards_total = 0.0
    env_eval = FraudEnvironment(PPOConfig(), seed=1234)

    for tx in txs:
        action, prob = agent.predict(tx.state, threshold)
        reward = env_eval.step(tx, action)
        ok     = tx.is_fraud == (action == 1)
        correct += ok
        rewards_total += reward
        print(f"${tx.amount:>8,.0f} | {tx.time_of_day:>5.2f} | {tx.velocity:>5.2f} | "
              f"{tx.geo_risk:>5.2f} | {tx.device_trust:>5.2f} | "
              f"{'FRAUD' if tx.is_fraud else 'LEGIT':>7} | {prob:>7.3f} | "
              f"{'BLOCK' if action == 1 else 'ALLOW':>6} | {reward:>+7.1f} | "
              f"{'✓' if ok else '✗':>4}")

    print("-" * 82)
    print(f"  Accuracy: {correct}/{n}  ({100*correct/n:.0f}%)  |  "
          f"Total reward: {rewards_total:+.1f}")
    print("=" * 82)


# ──────────────────────────────────────────────────────────────────────────────
# 11.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = PPOConfig(
        total_episodes = 400,
        batch_size     = 512,
        hidden         = 64,
        lr_actor       = 3e-4,
        lr_critic      = 1e-3,
        entropy_coef   = 0.03,
        seed           = 42,
    )

    agent, tracker, threshold = train(cfg)
    demo_inference(agent, threshold)
   
    
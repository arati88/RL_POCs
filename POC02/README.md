## POC-02: Q-Learning Optimization
## Policy Improvement in Taxi-v3 using Tabular Reinforcement Learning

## Objective

The goal of POC-02 is to implement and validate Q-Learning, a model-free Reinforcement Learning algorithm, to transform a random agent into a reward-maximizing policy.

This POC demonstrates:

How value functions are learned using the Bellman equation
How the Q-table evolves through experience
How exploration vs exploitation is handled (ε-greedy strategy)
How training improves long-term rewards
How RL behavior can be validated using BDD (Behavior Driven Development)
Unlike POC-01 (random agent), this experiment introduces learning through experience.


## From Random to Learning Agent
In POC-01:

The agent selected actions randomly
Rewards were highly negative (~ -200 average)
No learning occurred

In POC-02:

The agent updates Q-values after each interaction
The policy improves over time
Average reward increases significantly
Behavior becomes structured and goal-oriented

## Algorithm Used: Q-Learning

Q-Learning is a value-based, off-policy reinforcement learning algorithm.

It updates action values using the Bellman equation:
Q(s,a) = Q(s,a) + α [ r + γ max(Q(s′)) − Q(s,a) ]

Where:

s → current state
a → selected action
r → reward received
s′ → next state
α (alpha) → learning rate
γ (gamma) → discount factor

The agent iteratively improves its estimate of the optimal action-value function.

## Exploration Strategy: ε-Greedy

To balance learning and exploitation:
With probability ε → choose random action (explore)
With probability 1 − ε → choose best known action (exploit)

During training:

ε starts high (1.0)
ε gradually decays
Agent shifts from exploration to exploitation

This allows efficient discovery of optimal policies.

## Environment Used
Parameter	            Details
Environment	        Taxi-v3 (Gymnasium)
State Space	        500 discrete states
Action Space	    6 discrete actions
Reward Structure    +20 (success), -10 (illegal), -1 (step penalty)

Taxi-v3 remains ideal because:

It has discrete states (perfect for tabular Q-Learning)
It provides structured penalties and rewards
It demonstrates shortest-path optimization clearly

## Training Configuration
Parameter	            Value
Episodes	            1000
Learning Rate (α)	    0.1
Discount Factor (γ)	    0.9
Initial Epsilon	        1.0
Epsilon Decay	        0.995
Minimum Epsilon	        0.01

Training duration: ~3–4 seconds

## Behavior Driven Development (BDD) Validation

The following scenarios are tested:

Q-table initializes correctly (500 × 6)
Q-values update after a learning step
Epsilon decays properly
Average reward improves after training
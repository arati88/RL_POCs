## POC03 — Markov Decision Process (MDP) Modeling

## Objective
This module serves as the foundation for Dynamic Programming algorithms, including:

Value Iteration
Policy Iteration

By formally validating the MDP structure, we ensure correctness before implementing optimal control algorithms.

## Overview
This Proof of Concept (POC03) focuses on formally modeling the Taxi-v3 reinforcement learning environment as a Markov Decision Process (MDP).

The goal is to:

Extract the MDP components from the Gym environment
Validate the mathematical structure of the MDP
Ensure correctness using BDD (Behavior Driven Development)

## MDP Definition
An MDP is defined as a 5‑tuple:

S, A, P, R, γ

Where:

S → Set of states
A → Set of actions
P(s'|s,a) → Transition probability function
R(s,a) → Reward function
γ → Discount factor

## Taxi-v3 MDP Components

State Space (S)

Taxi-v3 consists of:

25 taxi positions (5×5 grid)
5 passenger states (4 pickup locations + inside taxi)
4 destination locations

Total states:

25 × 5 × 4 = 500
|S| = 500

## Action Space (A)

There are 6 discrete actions:

Action ID	Description
0	        Move South
1	        Move North
2	        Move East
3	        Move West
4	        Pickup passenger
5	        Dropoff passenger

|A| = 6

## Transition Function (P)

Taxi-v3 is deterministic.

For every (state, action) pair:
P(s'|s,a) = 1 for exactly one next state.
There is no stochasticity in movement.

## Deterministic MDP

Reward                          Function (R)
Situation	                     Reward
Valid dropoff	                  +20
Illegal pickup/dropoff	          -10
Any movement	                   -1

## BDD Validation

The following are validated using Behave:

State space size
Action space size
Determinism
Transition consistency
Reward structure correctness

Run tests using:
python -m behave

Expected result:
All scenarios passed.
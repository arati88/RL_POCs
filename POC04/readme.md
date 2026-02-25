## POC04 Policy Gradient Learning

## Overview

This project demonstrates a Policy Gradient (REINFORCE) Reinforcement Learning algorithm applied to a retail coupon optimization problem.

The goal is to train an RL agent to decide:

Give Coupon
Do Not Give Coupon

in order to maximize profit per customer interaction.

The solution includes:
    Synthetic dataset generation
    Custom RL environment
    Stochastic policy (Softmax)
    Reward-weighted gradient update
    Baseline strategy comparison
    Behavior-Driven Development (BDD) testing using Gherkin

## How it works

## 1. Synthetic Data Generation

Creates 100 simulated customers with:
Customer Type:
    coupon_sensitive
    normal
    neutral
Base purchase amount
Profit if:
    No coupon
    Coupon issued

## 2. Retail Environment

Custom RL environment:
    States → Customer types (3)
    Actions → {0: No Coupon, 1: Give Coupon}
    Reward → Profit from decision

## 3. Policy Gradient Learning
Implements:
    Parameterized policy (θ matrix)
    Softmax stochastic action selection
    Monte Carlo return computation
    Reward-weighted gradient ascent update

update rule:
    θ = θ + α * G * (one_hot(action) - policy_probs)

Where:
    α = Learning rate
    G = Discounted return
    policy_probs = Softmax probabilities

## 4. Training

Episodes: 3000
Steps per episode: 50
Discount factor (γ): 0.95
During training, the agent learns which action maximizes long-term profit.

## 5. Evaluation

The trained policy is compared against:
    Always Give Coupon
    Never Give Coupon
    RL Optimized Policy

Output Sample
===== STRATEGY COMPARISON =====

Strategy                  Average Profit per Step
RL Optimized Policy       207.868
Always Give Coupon        153.039
Never Give Coupon         188.854

## BDD Testing (Gherkin)

Feature file:
    Feature: Policy Gradient Learning for Retail Coupon Optimization

Test scenario validates:
    RL policy outperforms always-coupon strategy
    RL policy outperforms never-coupon strategy

Run Tests
    python -m behave

Example output
    1 feature passed, 0 failed, 0 skipped
    1 scenario passed, 0 failed, 0 skipped
    6 steps passed, 0 failed, 0 skipped





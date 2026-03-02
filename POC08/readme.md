# POC-08: Reward Curve Visualization & Convergence Analysis

# Objective

The primary objective of this POC is to:
    Demonstrate learning behavior of a PPO (Proximal Policy Optimization) agent through reward curve visualization and convergence analysis.
    
Rather than focusing only on classification accuracy, this project evaluates whether the reinforcement learning agent:
    Learns over time
    Improves its reward signal
    Stabilizes after sufficient training
    Exhibits convergence behavior

# Why Reward Curves Matter
In reinforcement learning, performance is measured by reward progression over episodes, not just final classification scores.

A well-trained agent should show:
    Increasing reward trend
    Reduced variance toward later episodes
    Stable reward plateau
    Positive final average reward
These properties indicate convergence.

# Implementation Overview
Environment
    Fraud detection framed as a binary action problem:
    Action 1 → Predict Fraud
    Action 0 → Predict Not Fraud
    Reward Design

Reward shaping reflects business risk:
    True Positive → High reward
    True Negative → Small reward
    False Positive → Moderate penalty
    False Negative → Heavy penalty
This makes the reward signal business-aligned.

# Reward Curve Generation

During training:
    Reward per episode is recorded
    Moving average smoothing is applied
    Final average reward is computed
    Variance in later episodes is analyzed
    The reward curve is saved


# Convergence Criteria

The model is considered Stable when:
    Final average reward is positive
    Final reward > initial reward
    Variance in last N episodes is low
    Reward trend is upward
These checks simulate production-level learning validation.

# BDD Testing
Instead of manually inspecting reward plots, we define executable behavioral rules using Gherkin syntax.

BDD ensures that:
    The PPO agent actually improves over time
    The reward curve demonstrates learning
    Convergence behavior is stable
    Business performance thresholds are satisfied
    This transforms reward visualization into automated validation logic.

BDD automatically verifies that:

 Reward improves over time
 Final reward is positive
 Convergence is stable
 Performance exceeds minimum business thresholds
 Regression in learning behavior is detected

If training degrades in future experiments, tests will fail immediately.

# Running BDD Tests
python -m behave

# Expected output:
1 feature passed
3 scenarios passed
All steps passed
If convergence degrades or metrics drop below thresholds, the test suite will fail.

# Conclusion
The primary outcome is not just a fraud classifier, but a validated learning process demonstrating stability and improvement over time.
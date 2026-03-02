Feature: PPO Fraud Detection using Bernoulli Policy
  The fraud engine should correctly learn to block fraud
  while minimizing customer friction.

  Background:
    Given a PPO fraud detection configuration
    And a fraud environment

  Scenario: Model trains without crashing
    When the PPO agent is trained for 5 episodes
    Then the training should complete successfully

  Scenario: Fraud probability is higher for fraudulent transactions
    Given the trained PPO agent
    When I evaluate 200 transactions
    Then average fraud probability for fraud cases should be higher than legit cases

  Scenario: Threshold tuning improves F1 score
    Given the trained PPO agent
    When I tune the decision threshold
    Then the best F1 score should be greater than 0.5

  Scenario: Fraud transactions should more often be blocked
    Given the trained PPO agent
    When I evaluate 300 transactions
    Then recall should be greater than 0.6

  Scenario: PPO policy update remains stable
    Given the trained PPO agent
    Then the KL divergence should be less than 0.2
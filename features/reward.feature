Feature: Reward Calculation
  In order to standardize reward logic
  As a reinforcement learning system
  I want to calculate rewards based on success status

  Scenario: Reward is positive when success is true
    When reward is calculated for success True
    Then the reward value should be 1

  Scenario: Reward is negative when success is false
    When reward is calculated for success False
    Then the reward value should be -1

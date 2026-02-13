Feature: RL Agent Behavior
  In order to act in an environment
  As an RL agent
  I want to initialize correctly, select actions, and update rewards

  Scenario: Agent initializes with zero reward
    Given a new RL agent
    Then the total reward should be 0

  Scenario: Agent selects a valid action
    Given a new RL agent
    When the agent selects an action for state 0
    Then the selected action should be either 0 or 1

  Scenario: Agent updates total reward
    Given a new RL agent
    When the agent receives reward 5
    Then the total reward should be 5

  Scenario: Agent accumulates rewards
    Given a new RL agent
    When the agent receives reward 3
    And the agent receives reward 2
    Then the total reward should be 5

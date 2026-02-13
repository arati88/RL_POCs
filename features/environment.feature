Feature: Environment Behavior
  In order to simulate an RL environment
  As a system component
  I want to generate states and return correct rewards

  Scenario: Environment initializes with valid state
    Given a new environment
    Then the state should be between 0 and 10

  Scenario: Environment gives positive reward for correct action
    Given a new environment with state 4
    When the agent takes action 0
    Then the reward should be 1
    And the episode should be done

  Scenario: Environment gives negative reward for incorrect action
    Given a new environment with state 5
    When the agent takes action 0
    Then the reward should be -1
    And the episode should be done

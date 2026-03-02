Feature: Taxi Environment Basic Interaction

  Scenario: Create Taxi environment
    Given a Taxi-v3 environment
    Then the action space size should be 6

  Scenario: Reset environment
    Given a Taxi-v3 environment
    When the environment is reset
    Then the initial state should be between 0 and 499

  Scenario: Take one random step
    Given a Taxi-v3 environment
    When the environment is reset
    And a random action is taken
    Then the returned state should be between 0 and 499
    And the reward should be an integer
    And the episode status should be valid

  Scenario: Run 5 random steps
    Given a Taxi-v3 environment
    When the environment is reset
    And 5 random actions are executed
    Then the total reward should be calculated

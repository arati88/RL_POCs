Feature: RL Decision Loop
  In order to complete one RL episode
  As a reinforcement learning system
  I want the agent and environment to interact correctly

  Scenario: Successful episode execution
    Given an environment with state 4
    And a new RL agent
    When the agent runs one episode
    Then the returned state should be 4
    And the action should be either 0 or 1
    And the reward should be either 1 or -1
    And the total reward should equal the reward

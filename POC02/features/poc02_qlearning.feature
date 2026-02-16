Feature: Q-Learning Optimization in Taxi Environment

  Scenario: Q-table should initialize correctly
    Given the Taxi environment is created
    When the Q-table is initialized
    Then the Q-table should have 500 rows and 6 columns

  Scenario: Q-values should update after one learning step
    Given a Q-table initialized with zeros
    When a learning update is performed
    Then the Q-value for the selected state-action should change

  Scenario: Epsilon should decay after an episode
    Given epsilon is initialized to 1.0
    When epsilon decay is applied
    Then epsilon should be less than 1.0

  Scenario: Agent reward should improve after training
    Given the agent is trained for multiple episodes
    Then the average reward of the last 100 episodes should show learning improvement

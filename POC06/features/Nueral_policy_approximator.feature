Feature: REINFORCE Trading Policy POC06
  Verify neural policy agent interacts correctly with the stock environment

  Background:
    Given a fresh stock trading environment
    And a fresh policy network

  Scenario: Policy outputs correct action logits
    When the policy network is queried with the current state
    Then it should return 3 action logits

  Scenario: BUY action executes correctly
    Given the agent has initial cash of 10000
    When the agent buys at price 100
    Then position should be 1
    And balance should decrease
    And reward should be zero

  Scenario: SELL action executes correctly
    Given the agent has bought stock at price 100
    When price increases to 110 and agent sells
    Then position should be 0
    And reward should be positive
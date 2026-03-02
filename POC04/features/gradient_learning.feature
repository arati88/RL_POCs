Feature: Retail Coupon Optimization using Policy Gradient

  As a retail analytics system
  I want to optimize coupon issuance decisions
  So that average profit per customer interaction is maximized


  Scenario: Environment initializes correctly
    Given a retail environment
    Then the environment should have 3 states
    And the environment should have 2 actions


  Scenario: Step function returns valid outputs
    Given a retail environment
    When I take action GiveCoupon
    Then the next state should be valid
    And the reward should be a float


  Scenario: Agent training runs without error
    Given a retail environment
    And a policy gradient agent
    When I train the agent for 100 episodes
    Then training should complete successfully


  Scenario: RL policy outperforms naive strategies
    Given a retail environment
    And a trained policy gradient agent
    When I evaluate all strategies
    Then the RL policy profit should be greater than always coupon
    And the RL policy profit should be greater than never coupon


  Scenario: Learned policy makes rational decisions
    Given a trained policy gradient agent
    When I inspect the learned policy
    Then NormalBuyer should prefer NoCoupon
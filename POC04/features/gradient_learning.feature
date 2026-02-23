Feature: Policy Gradient Learning for Retail Coupon Optimization

  Scenario: RL agent should learn better than baseline strategies
    Given a synthetic retail dataset is generated
    And a Retail Environment is created
    And a Policy Gradient agent is initialized
    When the agent is trained for 3000 episodes
    Then the RL optimized policy should achieve higher profit than always giving coupon
    And the RL optimized policy should achieve higher profit than never giving coupon
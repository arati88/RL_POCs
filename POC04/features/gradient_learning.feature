Feature: Policy Gradient Learning for Retail Coupon Optimization

  # ----------------------------------------------------------------
  # Sanity checks – space sizes and transition validity
  # ----------------------------------------------------------------

  Scenario: Environment has 3 customer states and 2 actions
    Given a Retail Environment is created with default config
    Then the state space size should be 3
    And the action space size should be 2

  Scenario: Transitions always stay within the 3 customer states
    Given a Retail Environment is created with default config
    When 200 random steps are executed
    Then all visited states should be within the valid state space

  # ----------------------------------------------------------------
  # Reward calculation – deterministic scenarios via seeded RNG
  # ----------------------------------------------------------------

  Scenario: Reward is zero when no purchase occurs
    Given a Retail Environment is created with seed 0 and always-no-purchase probabilities
    When a NoCoupon action is taken from Neutral state
    Then the reward should be 0.0

  Scenario: Reward equals margin when purchase occurs without coupon
    Given a Retail Environment is created with seed 0 and always-purchase probabilities
    When a NoCoupon action is taken from Neutral state
    Then the reward should equal the configured margin

  Scenario: Reward equals margin minus coupon cost when purchase occurs with coupon
    Given a Retail Environment is created with seed 0 and always-purchase probabilities
    When a GiveCoupon action is taken from Neutral state
    Then the reward should equal margin minus coupon cost

  # ----------------------------------------------------------------
  # Training quality – RL should outperform naive baselines
  # ----------------------------------------------------------------

  Scenario: RL agent should learn better than baseline strategies
    Given a Retail Environment is created with default config
    And a Policy Gradient agent is initialized
    When the agent is trained for 3000 episodes
    Then the RL optimized policy should achieve higher profit than always giving coupon
    And the RL optimized policy should achieve higher profit than never giving coupon

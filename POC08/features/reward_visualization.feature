Feature: PPO Fraud Detection Evaluation


  Scenario: Validate final model metrics
    Given the POC-08 results JSON file exists
    When I load the evaluation metrics
    Then precision should be greater than 0.70
    And recall should be greater than 0.55
    And F1 score should be greater than 0.65
    And accuracy should be greater than 0.90

  Scenario: Validate convergence behavior
    Given the POC-08 results JSON file exists
    When I check convergence analysis
    Then convergence status should be "Stable"
    And final average reward should be positive

  Scenario: Validate reward learning trend
    Given the POC-08 results JSON file exists
    When I analyze reward history
    Then the final average reward should be greater than the initial average reward
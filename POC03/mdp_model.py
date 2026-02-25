"""
POC-03: Markov Decision Process (MDP) Modeling
Environment: Taxi-v3 (Gymnasium)

This module formally represents the Taxi-v3 environment
as a Markov Decision Process (MDP).

An MDP is defined as a 5-tuple:

    (S, A, P, R, γ)

Where:
    S  -> Finite set of states
    A  -> Finite set of actions
    P  -> Transition probability function
           P(s' | s, a)
    R  -> Reward function
           R(s, a)
    γ  -> Discount factor (0 ≤ γ ≤ 1)

Taxi-v3 is:
    - Fully observable
    - Finite
    - Discrete
    - Deterministic

This file extracts and exposes those components explicitly.
"""

import gymnasium as gym


class TaxiMDP:
    """
    Formal MDP representation of the Taxi-v3 environment.

    This class extracts and exposes:

        • State space size (|S|)
        • Action space size (|A|)
        • Transition dynamics (P)
        • Reward function (R)
        • Discount factor (γ)

    The internal transition structure is obtained from:
        env.unwrapped.P

    Which stores:
        P[state][action] = [
            (probability, next_state, reward, done)
        ]
    """

    def __init__(self, gamma: float = 0.9):
        """
        Initialize the Taxi environment and extract MDP components.

        Parameters
        ----------
        gamma : float
            Discount factor for future rewards.
            Controls importance of long-term returns.

        Notes
        -----
        γ close to 1 → future rewards matter more
        γ close to 0 → agent focuses on immediate rewards
        """

        # Create Taxi-v3 environment
        self.env = gym.make("Taxi-v3")

        # Discount factor (part of MDP definition)
        self.gamma = gamma

        # ----------------------------------------------------
        # Extract State Space (S)
        # ----------------------------------------------------
        # Taxi-v3 has 500 discrete states:
        # 25 taxi positions × 5 passenger states × 4 destinations
        self.num_states = self.env.observation_space.n

        # ----------------------------------------------------
        # Extract Action Space (A)
        # ----------------------------------------------------
        # 6 discrete actions:
        # 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
        self.num_actions = self.env.action_space.n

        # ----------------------------------------------------
        # Extract Transition Function (P)
        # ----------------------------------------------------
        # Gym stores transition dynamics internally as:
        # P[state][action] → list of possible outcomes
        #
        # Each outcome is:
        # (probability, next_state, reward, done)
        #
        # Taxi-v3 is deterministic:
        # For each (state, action) there is exactly one outcome
        # with probability = 1.0
        self.P = self.env.unwrapped.P

    # ========================================================
    # MDP COMPONENT ACCESS METHODS
    # ========================================================

    def get_state_space_size(self) -> int:
        """
        Return the total number of states |S|.

        Returns
        -------
        int
            Number of discrete states.
        """
        return self.num_states

    def get_action_space_size(self) -> int:
        """
        Return the total number of actions |A|.

        Returns
        -------
        int
            Number of discrete actions.
        """
        return self.num_actions

    def get_transition(self, state: int, action: int):
        """
        Retrieve transition information for a given (state, action).

        Parameters
        ----------
        state : int
            Current state (s)
        action : int
            Selected action (a)

        Returns
        -------
        list of tuples
            [(probability, next_state, reward, done)]

        In deterministic environments:
            Length of list = 1
            Probability = 1.0
        """
        return self.P[state][action]

    def is_deterministic(self) -> bool:
        """
        Check whether the MDP is deterministic.

        A deterministic MDP satisfies:
            For each (s, a):
                - Exactly one possible next state
                - Transition probability = 1

        Returns
        -------
        bool
            True if deterministic, False otherwise.
        """

        for state in range(self.num_states):
            for action in range(self.num_actions):

                transitions = self.P[state][action]

                # Deterministic MDP must have exactly one outcome
                if len(transitions) != 1:
                    return False

                probability = transitions[0][0]

                # Probability must equal 1.0
                if probability != 1.0:
                    return False

        return True

    def sample_transition(self, state: int, action: int):
        """
        Return next_state, reward, done flag for a given (s, a).

        Since Taxi-v3 is deterministic,
        we directly extract the single outcome.

        Parameters
        ----------
        state : int
        action : int

        Returns
        -------
        next_state : int
        reward : float
        done : bool
        """

        probability, next_state, reward, done = self.P[state][action][0]
        return next_state, reward, done

    def close(self):
        """
        Close the environment to release resources.
        """
        self.env.close()


# ============================================================
# Manual Execution Block (For Debug / Verification)
# ============================================================

if __name__ == "__main__":

    # Instantiate MDP model
    mdp = TaxiMDP(gamma=0.9)

    print("---- MDP INFORMATION ----")
    print("Number of States (|S|):", mdp.get_state_space_size())
    print("Number of Actions (|A|):", mdp.get_action_space_size())
    print("Is Deterministic:", mdp.is_deterministic())

    # Demonstrate a sample transition
    example_state = 123
    example_action = 0  # South

    next_state, reward, done = mdp.sample_transition(
        example_state,
        example_action
    )

    print("\n---- SAMPLE TRANSITION ----")
    print(f"From state {example_state}, taking action {example_action}:")
    print("Next State:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    # Properly close environment
    mdp.close()

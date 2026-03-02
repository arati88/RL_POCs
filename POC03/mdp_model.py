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
    P  -> Transition probability function P(s' | s, a)
    R  -> Reward function R(s, a)
    γ  -> Discount factor (0 ≤ γ ≤ 1)

Taxi-v3 is:
    - Fully observable
    - Finite
    - Discrete
    - Deterministic
"""

import gymnasium as gym


class TaxiMDP:
    """
    Formal MDP representation of the Taxi-v3 environment.

    This class exposes:
        • State space size (|S|)
        • Action space size (|A|)
        • Transition dynamics (P)
        • Discount factor (γ)

    Transition structure is obtained from:
        env.unwrapped.P

    Which stores:
        P[state][action] = [
            (probability, next_state, reward, done)
        ]
    """

    def __init__(self, gamma: float = 0.9) -> None:
        """
        Initialize the Taxi environment and extract MDP components.

        Parameters
        ----------
        gamma : float
            Discount factor for future rewards.
        """
        self.env = gym.make("Taxi-v3")
        self.gamma = gamma

        # State space (|S|)
        self.num_states = self.env.observation_space.n

        # Action space (|A|)
        self.num_actions = self.env.action_space.n

        # Transition function (P)
        # Taxi-v3 is deterministic:
        # Each (state, action) has exactly one outcome
        self.P = self.env.unwrapped.P

    # ------------------------------------------------------------------
    # MDP COMPONENT ACCESS METHODS
    # ------------------------------------------------------------------

    def get_state_space_size(self) -> int:
        """Return total number of states |S|."""
        return self.num_states

    def get_action_space_size(self) -> int:
        """Return total number of actions |A|."""
        return self.num_actions

    def get_transition(self, state: int, action: int):
        """
        Retrieve transition information for a given (state, action).

        Returns
        -------
        list of tuples:
            [(probability, next_state, reward, done)]
        """
        return self.P[state][action]

    def is_deterministic(self) -> bool:
        """
        Check whether the MDP is deterministic.

        A deterministic MDP satisfies:
            - Exactly one possible next state
            - Transition probability = 1
        """
        for state in range(self.num_states):
            for action in range(self.num_actions):
                transitions = self.P[state][action]

                if len(transitions) != 1:
                    return False

                probability = transitions[0][0]
                if probability != 1.0:
                    return False

        return True

    def sample_transition(self, state: int, action: int):
        """
        Return next_state, reward, done for given (state, action).
        """
        probability, next_state, reward, done = (
            self.P[state][action][0]
        )
        return next_state, reward, done

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


# ----------------------------------------------------------------------
# Manual Execution Block (Debug / Verification)
# ----------------------------------------------------------------------

if __name__ == "__main__":

    mdp = TaxiMDP(gamma=0.9)

    print("---- MDP INFORMATION ----")
    print("Number of States (|S|):", mdp.get_state_space_size())
    print("Number of Actions (|A|):", mdp.get_action_space_size())
    print("Is Deterministic:", mdp.is_deterministic())

    example_state = 123
    example_action = 0  # South

    next_state, reward, done = mdp.sample_transition(
        example_state,
        example_action,
    )

    print("\n---- SAMPLE TRANSITION ----")
    print(
        f"From state {example_state}, "
        f"taking action {example_action}:"
    )
    print("Next State:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    mdp.close()
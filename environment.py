#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""

This module defines the Environment class used in
the POC-01 Reinforcement Learning decision loop.

The environment is responsible for:
- Generating an initial state
- Evaluating the agent's action
- Returning reward and episode status

"""

import random


class Environment:
    def __init__(self):  #Initialize the environment.

        #state (int): Randomly generated state between 0 and 10.
        self.state = self._generate_state()

    def _generate_state(self):
        return random.randint(0, 10)  #Returns a random integer between 0 and 10 (inclusive).

    def step(self, action):
       
        """
        Execute one interaction step in the environment.

        The rule:
            If the agent's action matches (state % 2),
            the action is considered correct (success).

        Args:
            action (int): Action selected by the agent (0 or 1).

        Returns:
            tuple:
                state (int): Current state.
                reward (int): +1 for correct action, -1 otherwise.
                done (bool): Episode termination flag (always True here).
        """
        reward = 1 if action == self.state % 2 else -1
        done = True  # One-step episode for simplicity
        return self.state, reward, done


# In[ ]:





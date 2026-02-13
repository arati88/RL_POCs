#!/usr/bin/env python
# coding: utf-8

# In[2]:
"""
This module defines the RLAgent class used in the
POC-01 Reinforcement Learning decision loop.

The agent is responsible for:
- Selecting an action based on the current state
- Maintaining and updating total accumulated reward

"""


import random


class RLAgent:        #A simple Reinforcement Learning agent.

    def __init__(self):    #Initialize the agent.
        self.total_reward = 0

    def select_action(self, state):        #Select an action based on the given state.
        """
        Select an action randomly for demonstration.
        """
        return random.choice([0, 1])

    def update_reward(self, reward):
        """
        Update total accumulated reward.
        """
        self.total_reward += reward


# In[ ]:





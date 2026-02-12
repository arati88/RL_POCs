#!/usr/bin/env python
# coding: utf-8

# In[2]:


# poc_01_decision_loop/agent.py

import random


class RLAgent:
    def __init__(self):
        self.total_reward = 0

    def select_action(self, state):
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





#!/usr/bin/env python
# coding: utf-8

# In[1]:


# poc_01_decision_loop/environment.py

import random


class Environment:
    def __init__(self):
        self.state = self._generate_state()

    def _generate_state(self):
        return random.randint(0, 10)

    def step(self, action):
        """
        Simulate environment transition.
        If action equals state % 2 → success.
        """
        reward = 1 if action == self.state % 2 else -1
        done = True  # One-step episode for simplicity
        return self.state, reward, done


# In[ ]:





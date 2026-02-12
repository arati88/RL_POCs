#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.getcwd()


# In[3]:


import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("."))  # if notebook is in root


# In[6]:


# poc_01_decision_loop/main.py

from agent import RLAgent
from environment import Environment


def run_episode():
    agent = RLAgent()
    env = Environment()

    state = env.state
    action = agent.select_action(state)
    _, reward, done = env.step(action)

    agent.update_reward(reward)

    return {
        "state": state,
        "action": action,
        "reward": reward,
        "total_reward": agent.total_reward,
    }


if __name__ == "__main__":
    result = run_episode()
    print("POC-01 Execution Result:")
    print(result)


# In[ ]:





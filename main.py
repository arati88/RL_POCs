#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.getcwd()


# In[3]:

"""
This module orchestrates the interaction between:
- RLAgent
- Environment

It simulates one complete episode consisting of:
1. Environment state retrieval
2. Agent action selection
3. Environment response
4. Reward update
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("."))  # if notebook is in root


# In[6]:

from agent import RLAgent
from environment import Environment


def run_episode():

    """
    Execute one full reinforcement learning episode.

    Flow:
        - Initialize agent and environment
        - Retrieve initial state
        - Agent selects action
        - Environment evaluates action
        - Agent updates total reward

    Returns:
        dict: Summary of episode containing:
            - state (int)
            - action (int)
            - reward (int)
            - total_reward (int)
    """

    # Initialize agent and environment
    agent = RLAgent()
    env = Environment()

     # Retrieve current environment state
    state = env.state

    # Agent selects an action based on state
    action = agent.select_action(state)

    # Environment processes the action
    _, reward, done = env.step(action)

    # Update agent's accumulated reward
    agent.update_reward(reward)


    # Return episode summary
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





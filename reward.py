#!/usr/bin/env python
# coding: utf-8

# In[1]:

"""
This module defines the reward calculation logic for
the POC-01 Reinforcement Learning decision loop.

The reward function abstracts the logic of assigning
numerical rewards based on success or failure.
"""


def calculate_reward(is_success):
    """
    Calculate reward based on success condition.

    This function encapsulates reward assignment logic
    so that it can be reused or modified independently
    from the environment or agent

    Args:
        is_success (bool): Indicates whether the agent's
                             action was successful.

    Returns:
        int:
            +1 if success is True
            -1 if success is False
    """
    
    return 1 if is_success else -1


# In[ ]:





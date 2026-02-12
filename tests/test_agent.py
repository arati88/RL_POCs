#!/usr/bin/env python
# coding: utf-8

# In[3]:


import unittest
from agent import RLAgent


class TestAgent(unittest.TestCase):

    def test_initial_reward(self):
        agent = RLAgent()
        self.assertEqual(agent.total_reward, 0)

    def test_reward_update(self):
        agent = RLAgent()
        agent.update_reward(5)
        self.assertEqual(agent.total_reward, 5)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)


# In[ ]:





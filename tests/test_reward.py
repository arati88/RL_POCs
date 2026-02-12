#!/usr/bin/env python
# coding: utf-8

# In[3]:


import unittest
from reward import calculate_reward


class TestReward(unittest.TestCase):

    def test_positive_reward(self):
        self.assertEqual(calculate_reward(True), 1)

    def test_negative_reward(self):
        self.assertEqual(calculate_reward(False), -1)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)


# In[ ]:





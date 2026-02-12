#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
from main import run_episode


class TestIntegration(unittest.TestCase):

    def test_full_episode_runs(self):
        result = run_episode()

        self.assertIn("state", result)
        self.assertIn("action", result)
        self.assertIn("reward", result)
        self.assertIn("total_reward", result)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)


# In[ ]:





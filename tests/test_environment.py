#!/usr/bin/env python
# coding: utf-8

# In[5]:


import unittest
from environment import Environment


class TestEnvironment(unittest.TestCase):

    def test_step_returns_values(self):
        env = Environment()
        state, reward, done = env.step(0)

        self.assertIsInstance(state, int)
        self.assertIn(reward, [-1, 1])
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)


# In[ ]:





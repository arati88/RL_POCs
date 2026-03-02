{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5036a1f",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "POC-01: Introduction to Reinforcement Learning\n",
    "Environment: Taxi-v3\n",
    "Agent: Random Action Agent\n",
    "\n",
    "This script demonstrates:\n",
    "- State space (0–499)\n",
    "- Action space (6 discrete actions)\n",
    "- Reward mechanism\n",
    "- Basic agent-environment interaction\n",
    "\"\"\"\n",
    "\n",
    "import gym\n",
    "\n",
    "\n",
    "def main():\n",
    "    \"\"\"Run a short random interaction with Taxi-v3 environment.\"\"\"\n",
    "\n",
    "    # Create environment\n",
    "    env = gym.make(\"Taxi-v3\")\n",
    "\n",
    "    # Reset environment\n",
    "    state, info = env.reset()\n",
    "    print(\"Initial State:\", state)\n",
    "\n",
    "    total_reward = 0\n",
    "\n",
    "    # Run for fixed number of steps\n",
    "    for step in range(5):\n",
    "\n",
    "        # Sample random action\n",
    "        action = env.action_space.sample()\n",
    "\n",
    "        # Perform action\n",
    "        state, reward, terminated, truncated, info = env.step(action)\n",
    "\n",
    "        total_reward += reward\n",
    "\n",
    "        print(f\"\\nStep {step + 1}\")\n",
    "        print(\"Action:\", action)\n",
    "        print(\"Next State:\", state)\n",
    "        print(\"Reward:\", reward)\n",
    "\n",
    "        if terminated or truncated:\n",
    "            print(\"\\nEpisode finished.\")\n",
    "            break\n",
    "\n",
    "    env.close()\n",
    "    print(\"\\nTotal Reward:\", total_reward)\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

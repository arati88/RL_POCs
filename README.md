# POC-01: Foundations of Reinforcement Learning  
## Agent–Environment Interaction using Taxi-v3

## Objective

The goal of POC-01 is to demonstrate the fundamental Reinforcement Learning (RL) interaction loop:

**Agent → Action → Environment → Reward → Next State**

This POC establishes:

- How an RL environment operates  
- How actions influence rewards  
- How state transitions occur  
- Why learning an optimal policy is necessary  
- The limitations of a random (non-learning) agent  

This experiment intentionally uses a **random policy** to highlight the inefficiency of non-optimized behavior.

---

## Core Reinforcement Learning Loop

At each time step:

1. The **Agent** observes the current **State**
2. The agent selects an **Action**
3. The **Environment** processes the action
4. The agent receives:
   - A **Reward**
   - The **Next State**
5. The cycle repeats until the episode terminates

This interaction forms the foundation of all Reinforcement Learning algorithms.

---

## Environment Used

| **Parameter** |** Details** |

| Environment | Taxi-v3 (Gymnasium) |
| Action Space | 6 discrete actions |
| Termination | Episode ends after successful drop-off |

### Why Taxi-v3?

Taxi-v3 is a classic benchmark RL environment because:

- It has clearly defined rules  
- It provides structured rewards and penalties  
- It demonstrates illegal action penalties  
- It encourages shortest-path optimization

- ##  Environment Description

In Taxi-v3:

- A taxi must **pick up a passenger**
- Transport them to the **correct destination**
- Drop them off successfully

- ### Reward Structure

- Successful Drop-off: **+20**
- Illegal Pickup/Dropoff: **-10**
- Every step taken: **-1**

This reward system encourages:

- Efficiency  
- Correct action selection  
- Minimal steps  

Reinforcement Learning algorithms aim to transform this random behavior into an optimal, reward-maximizing policy through experience and iterative improvement.



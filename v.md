4. **Dynamic Programming (DP):** * **Policy Iteration:** Policy Evaluation, Policy Improvement. * **Value Iteration:** Direct computation of optimal value function. * **Applicability and Limitations of DP:** When DP works and when it doesn't (curse of dimensionality).
5. **Monte Carlo (MC) Methods:** * **Monte Carlo Prediction:** Estimating value functions from episodes. * **Monte Carlo Control:** On-policy MC control (e.g., MC ES, First-visit MC). * **Exploration vs. Exploitation:** Epsilon-greedy policy, other exploration strategies.
6. **Temporal Difference (TD) Learning:** * **TD Prediction:** TD(0), SARSA, Q-learning. * **TD Control:** SARSA, Q-learning algorithms. * **Advantages of TD over MC:** Learning online, learning from incomplete episodes. * **Relationship between MC and TD:** Understanding their differences and similarities.
7. **Exploration-Exploitation Dilemma:** * **Common Exploration Strategies:** Epsilon-greedy, Upper Confidence Bound (UCB), Boltzmann Exploration (Softmax). * **Balancing Exploration and Exploitation:** Importance for effective learning.
---
**Phase 4: Deep Reinforcement Learning Algorithms & Techniques**
**Goal:** Learn and implement core Deep Reinforcement Learning algorithms, combining deep learning with reinforcement learning principles.
**Key Topics to Learn:**
1. **Function Approximation in RL:** 
* **Using Neural Networks as Value Function Approximators (Q-networks, V-networks).** 
* **Using Neural Networks as Policy Approximators (Policy Networks).** 
* **Deep Q-Networks (DQN):** 
* **Experience Replay:** Importance for stability and breaking correlations. 
* **Target Networks:** Stabilizing target values during training. 
* **DQN Algorithm details and implementation.** * **Variations of DQN:** Double DQN, Prioritized Experience Replay, Dueling DQN (brief overview). 
* **Policy Gradient Methods:** 
* **REINFORCE Algorithm:** Monte Carlo policy gradient. 
* **Actor-Critic Methods:** Combining policy and value function approximation. 
* **Advantage Actor-Critic (A2C) and Asynchronous Advantage Actor-Critic (A3C):** On-policy actor-critic algorithms. 
* **Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO):** Advanced policy gradient algorithms for stable and efficient learning. 
* **Deep Deterministic Policy Gradient (DDPG):** 
* **Deterministic Policy Gradients:** For continuous action spaces. 
* **DDPG Algorithm details and implementation.** 
* **Twin Delayed DDPG (TD3):** Addressing overestimation bias in DDPG. * **Soft Actor-Critic (SAC):** 
* **Maximum Entropy Reinforcement Learning:** Encouraging exploration androbust policies. 
* **SAC Algorithm details and implementation.**
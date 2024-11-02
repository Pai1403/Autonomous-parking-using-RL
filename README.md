# Autonomous-parking-using-RL( Grid World Concept)

This Python script simulates an autonomous vehicle parking in a grid-like parking lot using Q-learning, a reinforcement learning algorithm. The environment is visualized using pygame, enabling users to observe the agent as it navigates obstacles, seeks parking spots, and learns from trial and error to improve its parking efficiency over time.

Key Features:
**Grid-Based Parking Lot Environment**: A 5x5 grid with designated start positions, parking spots, and obstacles. Each cell represents a state in the environment.

**Q-Learning Agent**: Uses reinforcement learning with a Q-table to explore the environment, avoid obstacles, and efficiently reach the goal states (parking spots).

**Dynamic Pygame Visualization**: Real-time display of the parking process, showing the car’s movement, parking spots, and obstacles.

**Customizable Parameters**: Easily adjustable learning rate, discount factor, exploration rate, and decay rate for the Q-learning agent to optimize learning efficiency.

**Randomized Initialization Option**: Start positions, obstacles, and goal states can be randomized, making the simulation more dynamic and adaptable.


Requirements:
numpy: For Q-table management and numerical operations.

pygame: For graphical display of the simulation environment.

```Python
pip install numpy
pip install pygame
```

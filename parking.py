import numpy as np
import random
import pygame

# Define the parking lot environment
class ParkingLot:
    def __init__(self, size=5):
        self.size = size
        self.start_state = (0, 0)
        self.goal_states = [(4, 4), (4, 3)]  # Parking spots (goal)
        self.obstacles = [(1, 1), (2, 2), (3, 3)]  # Obstacles
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        row, col = self.state

        # Define actions: 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0:  # Move up
            row = max(0, row - 1)
        elif action == 1:  # Move down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Move left
            col = max(0, col - 1)
        elif action == 3:  # Move right
            col = min(self.size - 1, col + 1)
        
        new_state = (row, col)

        # Rewards and penalties
        if new_state in self.goal_states:
            reward = 10  # Reward for successfully parking
            done = True
        elif new_state in self.obstacles:
            reward = -10  # Penalty for hitting an obstacle
            done = True
        else:
            reward = -1  # Penalty for each step to encourage efficiency
            done = False
        
        self.state = new_state
        return new_state, reward, done

# Q-learning agent for autonomous parking
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.8, epsilon_decay=0.995):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, 4))  # (rows, cols, actions)
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Explore: random action
        else:
            row, col = state
            return np.argmax(self.q_table[row, col])  # Exploit: best action

    def update_q_table(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        best_next_action = np.argmax(self.q_table[next_row, next_col])
        td_target = reward + self.gamma * self.q_table[next_row, next_col, best_next_action]
        self.q_table[row, col, action] += self.alpha * (td_target - self.q_table[row, col, action])

    def train(self, episodes=200):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            # Decay epsilon after each episode
            self.epsilon *= self.epsilon_decay

# Pygame visualization
class ParkingSimulation:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        pygame.init()
        self.grid_size = 100
        self.screen = pygame.display.set_mode((self.env.size * self.grid_size, self.env.size * self.grid_size))
        self.colors = {
            'car': (0, 128, 255),
            'goal': (0, 255, 0),
            'obstacle': (255, 0, 0),
            'empty': (200, 200, 200)
        }

    def draw_grid(self):
        for row in range(self.env.size):
            for col in range(self.env.size):
                color = self.colors['empty']
                if (row, col) in self.env.goal_states:
                    color = self.colors['goal']
                elif (row, col) in self.env.obstacles:
                    color = self.colors['obstacle']
                pygame.draw.rect(self.screen, color, pygame.Rect(col * self.grid_size, row * self.grid_size, self.grid_size, self.grid_size))

    def draw_car(self):
        row, col = self.env.state
        pygame.draw.rect(self.screen, self.colors['car'], pygame.Rect(col * self.grid_size, row * self.grid_size, self.grid_size, self.grid_size))

    def run(self, episodes=100):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                action = self.agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.agent.update_q_table(state, action, reward, next_state)
                state = next_state

                # Redraw the grid and car for each step
                self.screen.fill((0, 0, 0))  # Background color
                self.draw_grid()
                self.draw_car()
                pygame.display.flip()
                pygame.time.wait(500)  # Delay to visualize the steps

            self.agent.epsilon *= self.agent.epsilon_decay
        pygame.quit()

# Main function to run the simulation
def main():
    # Create the parking lot environment
    env = ParkingLot()

    # Create the Q-Learning agent
    agent = QLearningAgent(env)

    # Train the agent (without visualization)
    agent.train(episodes=900)

    # Start the Pygame simulation
    sim = ParkingSimulation(env, agent)
    sim.run(episodes=10)

if __name__ == "__main__":
    main()

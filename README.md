# Super Mario Bros AI with Reinforcement Learning

This project implements an AI agent using reinforcement learning to play the classic game Super Mario Bros. The AI agent learns to navigate and complete levels in the game by interacting with the environment and receiving rewards.

## Prerequisites

- Python 3.7 or above
- OpenAI Gym
- NES Py
- Stable Baselines 3

## Installation

1. Clone the repository:
> https://github.com/BilalSardar009/Reinforcement-Learning-Mario
2. Install the required dependencies:
> pip install gym-super-mario-bros nes-py stable-baselines3 matplotlib

## Usage

To run the AI agent and watch it play Super Mario Bros, follow these steps:

1. Import the necessary libraries and set up the game environment.

```python
# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Set up the game environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
```
2. Create a flag for restarting the game and run the game loop.
```python
# Create a flag - restart or not
done = True

# Loop through each frame in the game
for step in range(1000000):
    # Start the game to begin with
    if done:
        env.reset()
    
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    
    # Show the game on the screen
    env.render()
    
# Close the game
env.close()
```
3. Train the AI model using Proximal Policy Optimization (PPO).
```python
# Import necessary libraries for training
import os
from stable_baselines3 import PPO

# Set checkpoint and log directories
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Create the PPO model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

# Train the model
model.learn(total_timesteps=1000000)

# Save the trained model
model.save('super-mario-rl-model')

```
4. Use the trained model to play the game.
```python
# Load the trained model
model = PPO.load('super-mario-rl-model')

# Start the game
state = env.reset()

# Loop through the game
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
   ```
## Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please create a new issue or submit a pull request. 

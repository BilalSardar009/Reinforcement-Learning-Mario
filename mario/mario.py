# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import gdata
# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT) #simplify movements only 7 to learn AI easyly


# Create a flag - restart or not
done = True
# Loop through each frame in the game
for step in range(1000000): 
    # Start the game to begin with 
    if done: 
        # Start the gamee
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()


##################################################

#model

##############################################

# # Import Frame Stacker Wrapper and GrayScaling Wrapper
# from gym.wrappers import GrayScaleObservation
# # Import Vectorization Wrappers
# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# # Import Matplotlib to show the impact of frame stacking
# from matplotlib import pyplot as plt


# # 1. Create the base environment
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# # 2. Simplify the controls 
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
# # 3. Grayscale
# env = GrayScaleObservation(env, keep_dim=True)
# # 4. Wrap inside the Dummy Environment
# env = DummyVecEnv([lambda: env])
# # 5. Stack the frames
# env = VecFrameStack(env, 4, channels_order='last')


# state = env.reset()

# # state, reward, done, info = env.step([env.action_space.sample()])
# # state, reward, done, info = env.step([env.action_space.sample()])
# # state, reward, done, info = env.step([env.action_space.sample()])

# # plt.figure(figsize=(20,16))
# # for idx in range(state.shape[3]):
# #     plt.subplot(1,4,idx+1)
# #     plt.imshow(state[0][:,:,idx])
# # plt.show()



# #########################################

# #Training


# ##########################################

# # Import os for file path management
# import os 
# # Import PPO for algos
# from stable_baselines3 import PPO
# # Import Base Callback for saving models
# from stable_baselines3.common.callbacks import BaseCallback




# CHECKPOINT_DIR = './train/'
# LOG_DIR = './logs/'


# # This is the AI model started
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
#             n_steps=512)

# # Train the AI model, this is where the AI model starts to learn
# model.learn(total_timesteps=1000000)

# model.save('thisisatestmodel')

# model = PPO.load('thisisatestmodel')

# # Start the game 
# state = env.reset()
# # Loop through the game
# while True: 
    
#     action, _ = model.predict(state)
#     state, reward, done, info = env.step(action)
#     env.render()
    
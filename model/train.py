import argparse
import os

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

from gymnasium.wrappers import AtariPreprocessing

import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='ALE/Pong-v5', choices=['ALE/Pong-v5'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Returns
    returns = {}

    # Initialize environment and config.
    env = gym.make(args.env, render_mode="rgb_array")
    env_config = ENV_CONFIGS[args.env]
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    
    # wrap the env in the record video
    env = gym.wrappers.RecordVideo(env=env, video_folder="./video3", name_prefix="test-video", episode_trigger=lambda x: x % 25 == 0)

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    
    # Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()  # Set the target network to evaluation mode

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    steps_done = 0  # Initialize a counter to track steps

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        print(episode)
        obs = preprocess(obs, env=args.env).unsqueeze(0).to(device)

        obs_stack_size = 4
        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)

      
        while not terminated:
            
            # Get action from DQN.
            action = dqn.act(obs_stack)

            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(dqn.convert_action[action.item()])

            # Preprocess incoming observation.
            next_obs = preprocess(next_obs, env=args.env).unsqueeze(0).to(device)
            next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)

            
            # Convert everything to PyTorch tensors
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([terminated], device=device)
            
            # Add the transition to the replay memory.
            memory.push(obs_stack, action, next_obs_stack, reward, done)

            obs_stack = next_obs_stack

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps_done % env_config["train_frequency"] == 0:
                # print(steps_done)
                optimize(dqn, target_dqn, memory, optimizer)
            
            # Update the target network every env_config["target_update_frequency"] steps.
            if steps_done % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            steps_done += 1  # Increment the steps counter

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            
            returns[episode+1] = mean_return
            
            with open('log_of_performance.json', 'w') as json_file:
                json.dump(returns, json_file, indent=4)

            # Ensure the directory exists before saving the model
            model_dir = f'models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = os.path.join(model_dir, f'{args.env}_best.pt')
            torch.save(dqn, model_path)

        
    # Close environment after training is completed.
    env.close()

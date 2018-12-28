from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import random
import argparse

from dqn_agent import Agent


def train(env, agent, brain_name, checkpoint_path, 
          n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        env (object): environment object
        agent (object): agent object
        brain_name (str): brain name for the env
        checkpoint_path (string): checkpoint path to save
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)
            break


def main(args):
    # initialize the env
    env = UnityEnvironment(file_name=args.env_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # size of state and action
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    # Initialize the agent 
    agent = Agent(state_size=state_size, action_size=action_size, args=args, seed=18)

    # Start the training
    train(env, agent, brain_name, args.checkpoint_path)

    # Close the env after the training
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drlnd Navigation Project",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_path', default="Banana.app", 
        help='environment name')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint_dqn.pth', 
        help='checkpoint path to save')
    parser.add_argument('--ddqn', action='store_true', help='whether to train Double DQN')
    parser.add_argument('--dueling', action='store_true', help='whether to train Dueling DQN')
    parser.add_argument('--prio', action='store_true', help='whether to prioritized replay buffer')

    args = parser.parse_args()

    main(args)

# Deep-Q-Network--Navigation
This is the first project of Udacity Deep Reinforcement Learning Nanodegree Program, which is an implementation of Deep Q Learning (DQN) navigating and collecting bananas in a large, square world.


## Demo

<p align="center"> 
<img src="demo.gif">
</p>

I've implement deep reinforcement learning algorithm with Pytorch. 
In this project, the following techniques have been implemented:

- [x] Deep Q Learning
- [x] Double Q Learning
- [x] Dueling Network
- [x] Prioritized Experience Play 


## Project Details

For this project, the agent must learn to move to as many yellow bananas as possible while avoiding blue bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
2. Place the file in this model directory, and unzip (or decompress) the file. 


## Instructions
To train an agent with DQN, simply run

`python main.py`

To activate advanced algorithm such as Double DQN (DDQN), Dueling Network or Prioritized Experience Replay, you can run

`python main.py --ddqn`

`python main.py --dueling`

`python main.py --prio`


## References
- [DQN Adventure: from Zero to State of the Art](https://github.com/higgsfield/RL-Adventure)
- [A tutorial of Prioritized Experience Replay by morvanzhou](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/)
- [DQN-DDQN-on-Space-Invaders by yilundu](https://github.com/yilundu/DQN-DDQN-on-Space-Invaders) 
- Framework provided by Udacity Deep Reinforcement Learning Nanodegree Program.
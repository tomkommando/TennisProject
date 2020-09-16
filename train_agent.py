import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from parameters import Params
from ddpg_agent1 import Agent

"""
Set parameters, see parameters.py
"""
params = Params() # instantiate the parameters

# load parameters
WEIGHTS_FOLDER = params.WEIGHTS_FOLDER
ACTOR_WEIGHTS = params.ACTOR_WEIGHTS
CRITIC_WEIGHTS = params.CRITIC_WEIGHTS
MU = params.MU
THETA = params.THETA
SIGMA = params.SIGMA
AGENT_SEED = params.AGENT_SEED
N_EPISODES = params.N_EPISODES
MAX_T = params.MAX_T
EPSILON = params.EPSILON_START
EPSILON_DECAY = params.EPSILON_DECAY

"""
Create Environment
"""
env = UnityEnvironment(file_name='./python/Tennis.exe', no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment, select if you want to train or not
env_info = env.reset(train_mode=True)[brain_name]
# env_info = env.reset(train_mode=False, )[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

"""
Train the agent
"""

agent0 = Agent(num_agents, state_size, action_size, AGENT_SEED, MU, THETA, SIGMA, ACTOR_WEIGHTS, CRITIC_WEIGHTS)
agent1 = Agent(num_agents, state_size, action_size, AGENT_SEED, MU, THETA, SIGMA, ACTOR_WEIGHTS, CRITIC_WEIGHTS)
agents = [agent0, agent1]

def ddpg(n_episodes=10, epsilon=0, eps_decay=1, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):

        for agent in agents:
            agent.reset()
        env_info = env.reset(train_mode=False)[brain_name]  # if you want to watch, set train_mode=False

        states = env_info.vector_observations  # get the current state (for each agent)
        score = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:

            for i, state in enumerate(states):
                agents[i].state = state # update states for agents

            if random.random() > epsilon:
                actions = [agent.act(add_noise=True) for agent in agents]  # pick actions
            else:
                actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
                actions = np.clip(actions, -1, 1)  # all actions between -1 and 1


            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished

            for i, agent in enumerate(agents): # take step (for each agent)
                agent.step(actions[i], rewards[i], next_states[i], dones[i])

            score += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break
            #print('Score (max over agents) from episode {}: {}'.format(t, np.max(scores)))
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        epsilon = epsilon * eps_decay

        print('\rEpisode {}\tAverage Max Score: {:.2f}, Epsilon: {:.4f}'.format(i_episode, np.max(score), epsilon), end="")
        if i_episode+1 % print_every == 0:
            print('\rEpisode {}\tAverage Max Score: {:.2f}, Epsilon: {:.4f}'.format(i_episode, np.mean(scores_deque, epsilon)))

        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Max Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_deque)))
            torch.save(agent0.actor_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_actor1_sol.pth')
            torch.save(agent0.critic_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_critic1_sol.pth')
            torch.save(agent1.actor_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_actor2_sol.pth')
            torch.save(agent1.critic_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_critic2_sol.pth')
            break

    return scores


# train agent
scores = ddpg(n_episodes=N_EPISODES, epsilon=EPSILON, eps_decay=EPSILON_DECAY)
env.close()

# plot scores
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

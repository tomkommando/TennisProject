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
params = Params()  # instantiate the parameters

# load parameters
WEIGHTS_FOLDER = params.WEIGHTS_FOLDER
CRITIC0_WEIGHTS = params.CRITIC0_WEIGHTS
ACTOR0_WEIGHTS = params.ACTOR0_WEIGHTS
CRITIC1_WEIGHTS = params.CRITIC1_WEIGHTS
ACTOR1_WEIGHTS = params.ACTOR1_WEIGHTS
MU = params.MU
THETA = params.THETA
SIGMA = params.SIGMA
AGENT_SEED = params.AGENT_SEED
N_EPISODES = params.N_EPISODES
MAX_T = params.MAX_T

"""
Create Environment
"""
# if you want to watch the agent train. Select no_graphics=False
env = UnityEnvironment(file_name='./python/Tennis.exe', no_graphics=True)
#env = UnityEnvironment(file_name='./python/Tennis.exe', no_graphics=False)

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

agent0 = Agent(num_agents, state_size, action_size, AGENT_SEED, MU, THETA, SIGMA, ACTOR0_WEIGHTS, CRITIC0_WEIGHTS)
agent1 = Agent(num_agents, state_size, action_size, AGENT_SEED, MU, THETA, SIGMA, ACTOR1_WEIGHTS, CRITIC1_WEIGHTS)
agent1.memory = agent0.memory  # use common replay memory


agents = [agent0, agent1]  # add agents to list for easy access


def ddpg(n_episodes, max_t=MAX_T, print_every=10):
    scores_deque = deque(maxlen=100)
    scores = []
    avg_scores = []
    steps = []
    best_score = -np.inf
    for i_episode in range(1, n_episodes + 1):

        for agent in agents:  # reset agent noise process (for each agent)
            agent.reset()

        env_info = env.reset(train_mode=True)[brain_name]  # if you want to watch, set train_mode=False

        states = env_info.vector_observations  # get the current state (for each agent)
        score = np.zeros(num_agents)  # initialize the score (for each agent)
        t = 0
        while True:

            for i, state in enumerate(states):
                agents[i].state = state  # update states for agents

            # force random actions in the beginning.
            # if you want to watch the agents playing, don't force randomness :)
            if i_episode >= 500:
                actions = [agent.act(add_noise=True) for agent in agents]  # select an action (for each agent)
            else:
                actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
                actions = np.clip(actions, -1, 1)  # all actions between -1 and 1

            env_info = env.step(actions)[brain_name]  # send all actions to environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished

            for i, agent in enumerate(agents):  # take step (for each agent)
                agent.step(actions[i], rewards[i], next_states[i], dones[i])

            score += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones): # or t == max_t:  # exit loop if episode finished
                break
            t += 1
        # print(f"Score (max over agents) from episode {i_episode}: {np.max(score)}, with {t} steps")
        scores_deque.append(np.max(score))
        avg_scores.append(np.mean(scores_deque))
        scores.append(np.max(score))
        steps.append(t)

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Max Score: {:.2f}\n'.format(i_episode, np.mean(scores_deque)))

        if 0.5 <= np.mean(scores_deque)  and np.mean(scores_deque) > best_score:
            print('\nNew Best Score at {:d} episodes!\tAverage Max Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_deque)))
            best_score = np.mean(scores_deque)
            torch.save(agent0.actor_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_actor0_best0.pth')
            torch.save(agent0.critic_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_critic0_best0.pth')
            torch.save(agent1.actor_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_actor1_best0.pth')
            torch.save(agent1.critic_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_critic1_best0.pth')

    return scores, avg_scores, steps


# train agent
scores, avg_scores, steps = ddpg(n_episodes=N_EPISODES)

env.close()
# torch.save(agent0.actor_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_actor0_final.pth')
# torch.save(agent0.critic_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_critic0_final.pth')
# torch.save(agent1.actor_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_actor1_final.pth')
# torch.save(agent1.critic_local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_critic1_final.pth')

# plot scores
fig, ax = plt.subplots(2, 1)
ax[0].plot(scores)
ax[0].plot(avg_scores)

ax[0].set_ylabel("Score")
ax[0].set_xlabel("Episode")

ax[1].plot(steps)
ax[1].set_ylabel("Steps")
ax[1].set_xlabel("Episode")

fig.tight_layout()
plt.savefig(WEIGHTS_FOLDER + 'scores0')
plt.show()

from unityagents import UnityEnvironment
import numpy as np
from agent.dqn_agent import Agent
from collections import deque
import torch

# please do not modify the line below
env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)

#
# Main hyperparams
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
fc_layer_size = 64
GAMMA = 0.99            # discount factor
TAU = 1e-3           # for soft update of target parameters
LR = 1e-4
# learning rate
UPDATE_EVERY = 2        # how often to update the network
start_eps = 0.01
eps = start_eps
min_eps = 0.01
eps_decay = 0.98

seed = 13

agent_type = "double_dqn" # vanilla_dqn, double_dqn
params = {
          "state_size": len(state),
          "action_size": action_size,
          "lr" : LR,
          "seed": seed,
          "fc_layer_size": fc_layer_size,
          "tau": TAU,
          "gamma": GAMMA,
          "batch_size": BATCH_SIZE,
          "update_every": UPDATE_EVERY,
          "buffer_size": BUFFER_SIZE,
          "type_dqn": agent_type
         }


agent_dqn = Agent(**params)

agent_dqn.qnetwork_online.load_state_dict(torch.load(f"trained_agents/checkpoint_{agent_type}.pth"))
agent_dqn.qnetwork_target.load_state_dict(torch.load(f"trained_agents/checkpoint_{agent_type}.pth"))

all_scores = []
scores = deque(maxlen=100)
sim_episodes = 1000
early_stop = True

for i_episode in range(10):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    score = 0

    while True:
        # get the state
        state = env_info.vector_observations[0]
        action = agent_dqn.act(state, eps)  # select an action
        action = action.astype(int)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]

        # see if episode has finished
        print(f"Reward: {reward} ", end='\r')
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    # Adjust e-greedy exploration for each episSode
    scores.append(score)
    all_scores.append([i_episode, scores])

    print(f"Episode {i_episode} and score: {score} and eps {round(eps,2)}", end='\r')

    if i_episode % 10 == 0:
        print(f"Episode number {i_episode} with  average score of {np.mean(scores)} and eps {round(eps,2)}")


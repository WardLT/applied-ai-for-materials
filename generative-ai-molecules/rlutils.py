"""Utility functions used for explaining reinforcement learning"""

from tqdm import tqdm
import pandas as pd
import numpy as np

state_cols = ['cart_position', 'cart_velocity', 'pole_angle', 'pole_velocity']


def enact_policy(env, agent):
    """Run a policy on an environment given an agent"""
    states = []
    rewards = []
    actions = []
    dones = []

    # Reset the system
    state = env.reset()
    done = False

    # Step until "done" flag is thrown
    while not done:
        action = agent.get_action(state)
        state, reward, done, data = env.step(action)  # Just push it to one side as an example
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        dones.append(done)

    states = pd.DataFrame(states, columns=state_cols)
    states['reward'] = rewards
    states['step'] = np.arange(len(states))
    states['action'] = actions
    states['done'] = dones
    return states

def evaluate_agent(env, agent, n_episodes, train=True):
    """Evalaute an agent over many episodes of the cart-pole game
    
    Args:
        env: Test environment
        agent: Agent to use and train
        n_episodes: Number of episodes to run with the game
        train: Whether to train the agent after each episode
    Returns:
        Dataframe with the results of each episode
    """
    
    # Storage for the length of each episode
    length = []
    
    # Run the desired number of episodes
    for i in tqdm(range(n_episodes), leave=False):
        # Run the environment
        states = enact_policy(env, agent)
        length.append(len(states))
        
        # Update agent, if desired
        if train:
            agent.train(states)
        
    # Make the output
    return pd.DataFrame({'length': length, 'episode': np.arange(n_episodes)})


class Agent:
    """Base class for an agent. 
    
    Defines the operations needed to use and train the agent.
    """
    
    def get_action(self, state):
        """Generate an action given the state of the system"""
        raise NotImplementedError()
    
    def train(self, states):
        """Update an agent's policies given some examples of states"""
        raise NotImplementedError()

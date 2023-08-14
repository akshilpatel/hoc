import numpy as np
from scipy.special import expit
from collections import defaultdict
from typing import Union, Tuple, List
from abc import ABC, abstractmethod

    

def run_epoch(agent, env, config: dict) -> None: 
    agent.set_train()
    obs, _ = env.reset()
    options = agent.choose_options(obs, lowest_level=1)
    
    
    for step in range(config["num_steps"]):
        action = agent.choose_action(obs, options)
        next_obs, reward, done, truncated, *_ = env.step(action)

        lowest_term, term_probs = agent.get_lowest_terminations(next_obs, options) # Note lowest means furthest from primitive. 
        next_options, optional_stats = agent.choose_options(next_obs, lowest_level=lowest_term)
        transition = (obs, options, action, reward, next_obs, done, next_options)

        agent.process_transition(transition, term_probs)
        agent.update(transition, term_probs)

        if done or truncated:
            obs, _ = env.reset()
            options = agent.choose_options(obs, levels=range(1, agent.num_levels))
        else:
            obs = next_obs
            options = next_options



def run_eval_episode(agent, env):
    obs, _ = env.reset()
    agent.set_eval()
    options = agent.choose_options(obs, levels=range(1, agent.num_levels)) 
    all_transitions = []
    
    while True:
        action = agent.choose_action(obs, options)
        next_obs, reward, done, truncated, *_ = env.step(action)

        lowest_term, term_probs = agent.get_lowest_terminations(next_obs, options)
        next_options, optional_stats = agent.choose_options(next_obs, range(lowest_term, agent.num_levels))
        
        transition = (obs, options, action, reward, next_obs, done, next_options)
        agent.process_transition(transition, term_probs)
        agent.update(transition, term_probs)

        all_transitions.append((*transition, term_probs))

        if done or truncated:
            break
        else:
            obs = next_obs
            options = next_options
    
    return all_transitions
    

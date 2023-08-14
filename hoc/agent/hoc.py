import numpy as np 
import functools
import operator
from typing import Union, List, Tuple, Optional, Dict
from collections import defaultdict
from scipy.special import expit, logsumexp
from gym import spaces


def make_hashable(x: Union[int, float, np.ndarray, list, tuple]) -> tuple:
    """Converts the input into a hashable for use as a key

    Args:
        x (Union[int, float, np.ndarray, list, tuple]): Raw input to be converted.

    Raises:
        NotImplementedError: If the input is not a supported type.

    Returns:
        tuple: Hashable version of the input.
    """
    if isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, list):
        return tuple(functools.reduce(operator.concat, x))
    elif isinstance(x, tuple):
        return x
    else:
        raise NotImplementedError

class IntraOptionQTable:
    def __init__(self, discount: float, lr: float, level: int) -> None:
        self.lr = lr
        self.discount = discount
        self.level = level
        self.table = defaultdict(float) # Input will be obs and option_chain and option to get q value
        

    def _preprocess_obs(self, obs: Union[int, float, np.ndarray, list, tuple]) -> tuple:
        """
        Converts the input into a hashable for use as a key in the `weights` dictionary. 
        Assumes single observation is given.
        Args:
            obs (Union[np.ndarray, list, tuple]): Observation from environment.

        Returns:
            tuple: Outputs a flat tuple 
        """
        return make_hashable(obs)

    def get_q_value(
            self, 
            obs: Union[int, np.ndarray, list, tuple], 
            option_chain: Tuple[int],
            option: int
            ) -> Union[int, float, np.ndarray]:
        """Main API for calling the QTable. Note that for tabular methods, we assume single inputs.

        Args:
            obs (Union[int, np.ndarray, list, tuple]): Single observation.
            option_chain (Tuple[int], optional): Tuple of options executing above this critic's level. Length = self.level-2

        Returns:
            Union[int, float]: Q(s, o^{1:l}) or Q(s, :) if no action is given. Shape should be at least 2d
        """
        obs_ = self._preprocess_obs(obs)
        
        q = self.weights[(obs_, option_chain, option)]
        return np.atleast_2d(q)

    def set_q_value(
            self, 
            obs: Union[int, np.ndarray, list, tuple], 
            option_chain: Tuple[int], 
            option: int,
            target: Union[float, int]) -> None:
        obs_ = self._preprocess_obs(obs)
        self.weights[(obs_, option_chain, option)] = target

    def update(self, transition: Dict[str, Union[int, float, bool]]) -> None:
        obs = transition["obs"]
        action = transition["actions"]
        reward = transition["task_rewards"]
        done = transition["dones"]
        next_obs = transition["next_obs"]

        # One-step update target
        next_obs_action_vals = self.get_q_value(next_obs)
        next_obs_val = np.max(next_obs_action_vals, axis=1)
        update_target = reward + (1-done) * self.discount * next_obs_val
        new_q = self.get_q_value(obs, action) * \
            (1 - self.lr) + self.lr * update_target
        self.set_q_value(obs, action, new_q)

class OptionQTable:
    pass

class OptionActionQTable: # Q(s, o, a)
    def __init__(self, discount, lr, num_obs, num_actions, num_options, reward_type):
        self.lr = lr
        self.discount = discount
        self.weights = np.zeros((num_obs, num_options, num_actions), dtype=np.float32)
        self.reward_type = reward_type
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_options = num_options

    def get_q_value(self, obs, option, action=None):
        
        if isinstance(obs, float) or isinstance(option, float) or isinstance(action, float):
            raise ValueError("Obs, options and actions must be integers or arrays of integers.") 
        if not isinstance(obs, int):
            obs = np.array(obs, dtype=int).reshape(-1,)
        if not isinstance(option, int):
            option = np.array(option, dtype=int).reshape(-1,)
        
        if action is None:
            q = self.weights[obs, option].reshape(-1, self.num_actions)
        else:
            # If it's a float array, a float, or a list or tuple.
            if not isinstance(action, int):
                action = np.array(action, dtype=int).reshape(-1,)

            q = self.weights[obs, option, action].reshape(-1, 1)
        return q # (batch_size, 1 or num_actions)

    def set_q_value(
        self, 
        obs:  Union[int, List, np.ndarray],
        option: Union[int, List, np.ndarray],
        action: Union[int, List, np.ndarray],
        target: Union[float, List, np.ndarray]
        ):
        """Setting Q values. This works for single or sequences of (state, action, target) """
        if isinstance(obs, float) or isinstance(option, float) or isinstance(action, float):
            raise ValueError("Obs, options and actions should be integers or sequences of integers.")
        
        if not isinstance(obs, int):
            obs = np.array(obs, dtype=int).reshape(-1,)
            
            if not isinstance(option, int):
                option = np.array(option, dtype=int).reshape(-1,)
            if not isinstance(action, int):
                action = np.array(action, dtype=int).reshape(-1,)
            
            if not isinstance(target, (int, float)):
                target = np.array(target).reshape(-1,)
        
            if len(obs) != len(option) or len(obs) != len(action):
                raise ValueError("The number of observations must equal the number of options and actions")
            elif len(obs) != len(target):
                raise ValueError("The number of observations must equal the number of targets")
                    
        self.weights[obs, option, action] = target
        return target

    @staticmethod
    def compute_vua(
        option,                 # (batch_size, 1)
        next_obs_val,           # (batch_size, 1)
        next_obs_option_vals,   # (batch_size, num_options)
        next_obs_option_beta,   # (batch_size, 1) but how to compute this? I would need beta_r(s')[option] 
        ):
        option_vals = next_obs_option_vals[range(next_obs_option_vals.shape[0]), option.reshape(-1,)].reshape(-1, 1)
        value_upon_arrival = (1.-next_obs_option_beta) * option_vals + next_obs_option_beta * next_obs_val 
        
        return value_upon_arrival # (batch_size, 1)

    def update(
        self, 
        batch,
        next_obs_val,           # (batch_size, 1)
        next_obs_option_vals,   # (batch_size, num_options)
        next_obs_option_beta    # batch_size, 1
        ):

        obs      = batch["obs"]
        option   = batch["options"]
        action   = batch["actions"]
        reward   = batch[self.reward_type + "_rewards"]
        done     = batch["dones"]

        # One-step update target
        vua = self.compute_vua(option, next_obs_val, next_obs_option_vals, next_obs_option_beta)
        update_target = reward + (1-done) * self.discount * vua

        # Update values upon arrival if desired
        old_q = self.get_q_value(obs, option, action)
        new_q = old_q + self.lr * (update_target - old_q)
        self.set_q_value(obs, option, action, new_q)

        return update_target - old_q


class OptionQTable:
    def __init__(self, discount, lr, num_obs, num_options, reward_type):
        self.lr = lr
        self.discount = discount
        self.weights = np.zeros((num_obs, num_options), dtype=np.float32)
        self.reward_type = reward_type
        self.num_obs = num_obs
        self.num_options = num_options

    def get_q_value(self, obs, option=None):
        if isinstance(obs, float) or isinstance(option, float):
                raise ValueError("Obs or options given are floats, they should be integers or arrays of integers.") 

        if not isinstance(obs, int):
            obs = np.array(obs, dtype=int).reshape(-1,)
                            
        if option is None:
            q = self.weights[obs].reshape(-1, self.num_options)
        else:
            # If it's a float array, a float, or a list or tuple.
            if not isinstance(option, int):
                option = np.array(option, dtype=int).reshape(-1,)

            q = self.weights[obs, option].reshape(-1, 1)
        return q

    def set_q_value(
        self, 
        obs:  Union[int, List, np.ndarray],
        option: Union[int, List, np.ndarray],
        target: Union[float, List, np.ndarray]
        ):
        """Setting Q values. This works for single or sequences of (state, option, target) """
        if isinstance(obs, float) or isinstance(option, float):
            raise ValueError("Obs and option are floats and should be integers.") 
        
        if not isinstance(obs, int):
            obs = np.array(obs, dtype=int).reshape(-1,)

        if not isinstance(option, int):
            option = np.array(option, dtype=int).reshape(-1,)
        
        if not isinstance(target, (int, float)):
            target = np.array(target).reshape(-1,)
        
        # Dealing with sequences
        if not isinstance(obs, int):
            if len(obs) != len(option):
                raise ValueError("The number of observations must equal the number of options")
            elif len(obs) != len(target):
                raise ValueError("The number of observations must equal the number of targets")
                    
        self.weights[obs, option] = target
        return target

   

class EgreedyPolicy:
    def __init__(self, rng, critic, epsilon, option_id):
        """Action Selecion implementation. Created to work with a Q function.

        Args:
            rng (np.random.Generator): _description_
            critic (QTable): (Should work with option or OptionAction Q table)
            epsilon (float): 
            id (int): _Index of Corresponding Option (either meta or policy over primitives)
        """
        self.rng = rng
        self.epsilon = epsilon
        self.critic = critic
        self.num_actions = critic.weights.shape[-1]
        self.reward_type = critic.reward_type
        self.option_id = option_id

    def sample(self, obs, q_vals, deterministic=False):
        """Does not work for batches.

        Args:
            obs (_type_): _description_
            q_vals (np.ndarray): (batch_size, num_actions)
            deterministic (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if deterministic or self.rng.uniform() > self.epsilon:
            action = self.rng.choice(np.flatnonzero(q_vals == q_vals.max(axis=-1)))
        else:
            action = self.rng.integers(0, q_vals.shape[-1])
        return action

    def get_prob(self, obs, q_vals, action=None):
        """Return prob for target policy."""
        q_max = np.max(q_vals, axis=-1).reshape(-1, 1)
        greedy_mask = (q_vals == q_max)
        out = np.zeros_like(q_vals)
        
        out[greedy_mask] = 1 # TODO:If this is used to mask as importance sampling weight, it will be wrong.
        out /= out.sum(axis=1).reshape(-1, 1)
        return out

class SoftmaxPolicy:
    def __init__(self, rng, critic, option_id, temp=1.):
        self.rng = rng
        self.critic = critic # Make sure this is the correct level of critic!
        self.temp = temp
        self.num_actions = critic.weights.shape[-1]
        self.option_id = option_id
        
    def get_prob(self, obs, option_chain, q_vals):
        q_max = q_vals.max(axis=-1).reshape(-1, 1)
        v = np.exp(q_vals - q_max)
        prob = v / v.sum(axis=-1).reshape(-1, 1)
        return np.array(prob).reshape(-1, self.num_actions)

    def sample(self, obs, q_vals, deterministic=False):
        prob = self.prob(obs, q_vals)
        if deterministic:
            return self.rng.choice(np.where(prob == prob.max(axis=-1)[1]))
        else:
            return self.rng.choice(self.num_actions, p=prob.squeeze())

class SigmoidTermination:
    """ 
        This class implements a level-wide sigmoid termination function. 
        One per level in the option hierarchy.
    """
    def __init__(self, rng, lr, discount, level):
        self.rng = rng
        self.weights = defaultdict(float(0.5)) # Input should be (obs, option_chain[:level], option) 
        self.level = level
        self.lr = lr
        self.discount = discount

    def get_prob(self, obs, option_chain: Tuple[int], option:int):
        """ Get the prob of termination for the given option.
        Args:
            obs (Union[int, float, np.ndarray]): Observation from environment.
            option_chain (Tuple[int]): Options currently executing.
            option (int): Option at current level, for which to query termination.

        Returns:
            prob (float): Probability of termination.
        """
        if isinstance(obs, (int, float)):
            obs = np.array(obs).reshape(1, 1)
        prob = expit(self.weights[(np.rint(obs).astype(int), option_chain, option)])
        return prob
    
    def sample(self, obs, option_chain, option, training_mode=True):
        """Query whether the option should terminate.

        Args:
            obs (Union[int, float, np.ndarray]): Observation from environment.
            option_chain (Tuple[int]): Options currently executing.
            option (int): Option at current level, for which to query termination.
            training_mode (bool, optional): Determines whether to sample probabilistically. Defaults to True.

        Returns:
            term (int): 1 if option should terminate, 0 otherwise.
            prob (float): Probability of termination.
        """
        prob = self.get_prob(obs, option_chain, option)
        if not training_mode:
            term = int(prob > 0.5)
        else:
            term = int(self.rng.uniform() < prob)

        return term, prob
        
    # def get_grad(self, obs, option_chain, option):
    #     t_prob = self.prob(obs)
    #     grad = t_prob * (1 - t_prob)
    #     return grad


class TabularHOCAgent:
    def __init__(self, 
                 num_options_per_level: Union[Tuple[int], List[int]],
                 rng: np.random.Generator
                 ) -> None:
        # Number of option levels in the hierarchy including root but not primitives.
        self.num_o_levels = 1 + len(num_options_per_level) 
        
        # List of lists of policies. Each list is a level in the hierarchy. Each level is a list of option policies.
        self.policies_by_level = [] 

        # One termination function per level in the hierarchy, apart from root option, which never terminates and primitives.
        self.termination_fns = [None] 
        
        # One critic per level in the hierarchy including one for primitives.
        self.critics = []
        self.rng = rng

    def choose_options(self, obs: Union[int, float, np.ndarray], option_chain: Tuple[int], lowest_level:int) -> Tuple[int]:
        """
        Main method for option selection. Called at each timestep to determine which option to execute. 
        The method does not always change the option chain.
        Args:
            obs (Union[int, float, np.ndarray]): Raw observation from environment.
            option_chain (Tuple[int]): Options currently being executed. length should be (self.num_o_levels - 1).  
            lowest_level (int): The levels above this will remain unchanged. lowest_level option and those below will be sampled.

        Returns:
            Tuple[int]: Next options to execute. This may be the same as the current option chain.
        """    
        if lowest_level == self.num_o_levels:
            return option_chain
        else:
            next_option_chain = list(option_chain)
            level = lowest_level
            while level < self.num_o_levels:
                policy = self.policies_by_level[level - 1]
                option = policy.sample(obs, option_chain[:level], self.training_mode) # DOes the input need to be the options from all levels above?
                next_option_chain[level] = option

        self.curr_option_chain = next_option_chain
        return next_option_chain
    
    def choose_action(self, obs:Union[int, float, np.ndarray], option_chain: Tuple[int]) -> int:
        return self.policies_by_level[-1][option_chain[-1]].sample(obs, option_chain[:-1], self.training_mode)
    
    def get_lowest_termination(self, obs, option_chain:Tuple[int]):
        """Queries the termination function at each level of the hierarchy to get the probabilities of termination.
        Args:
            obs (Union[int, float, np.ndarray]): Raw observation from environment.
            option_chain (Tuple[int]): Options currently executing, for which we query terminations.

        Returns:
            lowest_term (int): Furthest level from primitives, in the hierarchy, where the option has just terminated.
            term_probs (Tuple[float]): Probability of termination at each level of the hierarchy. Recorded for updating.
        """

        lowest_term = self.num_o_levels
        term_probs = np.zeros(self.num_o_levels, dtype=np.float32)

        for level in range(1, self.num_o_levels): # Skip root option
            term, term_prob = self.termination_fns[level].sample(
                                                            obs,
                                                            option_chain[:level],
                                                            option_chain[level],
                                                            self.training_mode,
                                                            return_prob=True
                                                            )
            term_probs.append(term_prob)
            if term:
                lowest_term = level           
            
        return lowest_term, term_probs
    
    def process_transition(self, transition, term_probs):
        pass

    def update_critics(self, transition, term_probs):
        pass

    def update_policies(self, transition, term_probs):
        pass
    
    def update_terminations(self, transition, term_probs):
        pass

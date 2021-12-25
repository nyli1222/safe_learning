from functools import update_wrapper
from collections import Sequence
import torch
import numpy as np
from utilities import batchify
from configuration import config

class SafeLearning():
    
    def __init__(self, discretization, lyapunov_function, dynamics,
                 lipschitz_dynamics, lipschitz_lyapunov,
                 tau, policy, beta=2., initial_set=None, adaptive=False):
        
        super(SafeLearning, self).__init__()

        self.discretization = discretization
        self.policy = policy

        # Keep track of the safe sets
        self.safe_set = np.zeros(np.prod(discretization.num_points),
                                 dtype=bool)

        self.initial_safe_set = initial_set
        if initial_set is not None:
            self.safe_set[initial_set] = True

        self.beta = beta

        # Discretization constant
        self.tau = tau

        # Make sure dynamics are of standard framework
        self.dynamics = dynamics

        # Make sure Lyapunov fits into standard framework
        self.lyapunov_function = lyapunov_function

        
        # Lyapunov values
        self.values = None

        self._lipschitz_dynamics = lipschitz_dynamics
        self._lipschitz_lyapunov = lipschitz_lyapunov

        #self.discretization_points = torch.tensor(self.discretization.all_points, dtype=config.dtype)

        self.update_values()

        # not sure what does refinement does
        self._refinement = np.zeros(discretization.nindex, dtype=int)
        if initial_set is not None:
            self._refinement[initial_set] = 1

    def get_safe_sample(self, perturbations=None, limits=None, positive=False,
                    num_samples=None, actions=None):
    
        """
        This function returns the most uncertain state-action pair close to the
        current policy (as a result of the perturbations) that is safe (maps
        back into the region of attraction).
        Parameters
        ----------
        lyapunov : instance of `Lyapunov'
            A Lyapunov instance with an up-to-date safe set.
        perturbations : ndarray
            An array that, on each row, has a perturbation that is added to the
            baseline policy in `lyapunov.policy`.
        limits : ndarray, optional
            The actuator limits. Of the form [(u_1_min, u_1_max), (u_2_min,..)...].
            If provided, state-action pairs are clipped to ensure the limits.
        positive : bool
            Whether the Lyapunov function is positive-definite (radially
            increasing). If not, additional checks are carried out to ensure
            safety of samples.
        num_samples : int, optional
            Number of samples to select (uniformly at random) from the safe
            states within lyapunov.discretization as testing points.
        actions : ndarray
            A list of actions to evaluate for each state. Ignored if perturbations
            is not None.
        Returns
        -------
        state-action : ndarray
            A row-vector that contains a safe state-action pair that is
            promising for obtaining future observations.
        var : float
            The uncertainty remaining at this state.
        """
        # Subsample from all safe states within the discretization
        safe_idx = np.where(self.safe_set)
        safe_states = self.discretization.index_to_state(safe_idx)
        if num_samples is not None and len(safe_states) > num_samples:
            idx = np.random.choice(len(safe_states), num_samples, replace=True)
            safe_states = safe_states[idx]

        # Generate state-action pairs around the current policy
        safe_states = torch.tensor(safe_states, dtype=config.dtype)
        actions = self.policy(safe_states).detach()

        
        state_actions = perturb_actions(safe_states,
                                        actions,
                                        perturbations=perturbations,
                                        limits=limits)
        
    
        next_states, variances = self.dynamics(state_actions)
        
        # Todo local lipschitz
        lv = self.lipschitz_lyapunov(next_states)
        std_sum = torch.sum(variances.sqrt(), axis=1, keepdims=True)
        upper_bound = lv * std_sum * self.beta
        next_states_values = self.lyapunov_function(next_states)

        # Check whether the value is below c_max
        future_values = next_states_values + upper_bound
        maps_inside = torch.less(future_values, self.c_max)

        maps_inside = maps_inside.squeeze(axis=1).detach().numpy()
        state_actions = state_actions.detach().numpy()
        upper_bound = upper_bound.detach().numpy()

        # Check whether states map back to the safe set in expectation
        if not positive:
            next_state_index = self.discretization.state_to_index(next_states.detach().numpy())
            safe_in_expectation = self.safe_set[next_state_index]
            maps_inside &= safe_in_expectation

        #Todo if nothing is safe
        bound_safe = upper_bound[maps_inside]
        max_id = np.argmax(bound_safe)
        max_bound = bound_safe[max_id].squeeze()
        return state_actions[maps_inside, :][[max_id]], max_bound


    def lipschitz_lyapunov(self, states):
        """Return the local Lipschitz constant at a given state.
        Parameters
        ----------
        states : ndarray or Tensor
        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_lyapunov is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_lyapunov, '__call__'):
            return self._lipschitz_lyapunov(states)
        else:
            return self._lipschitz_lyapunov

    def lipschitz_dynamics(self, states):
        """Return the Lipschitz constant for given states and actions.
        Parameters
        ----------
        states : ndarray or Tensor
        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_dynamics is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_dynamics, '__call__'):
            return self._lipschitz_dynamics(states)
        else:
            return self._lipschitz_dynamics

    def update_values(self):
        """Update the discretized values when the Lyapunov function changes."""
        self.values = self.lyapunov_function(self.discretization.all_points).squeeze()


    def v_decrease_confidence(self, states, next_states):
        """Compute confidence intervals for the decrease along Lyapunov function.
        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.
        Returns
        -------
        mean : np.array
            The expected decrease in values at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point
        """
        if isinstance(next_states, Sequence):
            next_states, error_bounds = next_states
            lv = self.lipschitz_lyapunov(next_states)
            bound = torch.sum(lv * torch.sqrt(error_bounds) * self.beta, dim=1, keepdims=True)

        else:
          bound = torch.tensor(0., dtype=config.dtype)

        v_decrease = (self.lyapunov_function(next_states)
                      - self.lyapunov_function(states))

        return v_decrease, bound


    def v_decrease_bound(self, states, next_states):
        v_decrease, v_upper_bound = self.v_decrease_confidence(states, next_states)
        return v_decrease + v_upper_bound


    def threshold(self, states, tau=None):
          
        """Return the safety threshold for the Lyapunov condition.
        Parameters
        ----------
        states : ndarray or Tensor
        tau : float or Tensor, optional
            Discretization constant to consider.
        Returns
        -------
        lipschitz : float, ndarray or Tensor
            Either the scalar threshold or local thresholds, depending on
            whether lipschitz_lyapunov and lipschitz_dynamics are local or not.
        """
        if tau is None:
            tau = self.tau
        lv = self.lipschitz_lyapunov(states)
      
        lf = self.lipschitz_dynamics(states).detach()
        return - lv * (1. + lf) * tau


    def update_safe_set(self):
    
        safe_set = np.zeros_like(self.safe_set, dtype=bool)
        refinement = np.zeros_like(self._refinement, dtype=int)
        if self.initial_safe_set is not None:
            safe_set[self.initial_safe_set] = True
            refinement[self.initial_safe_set] = 1
        value_order = np.argsort(self.values)
        safe_set = safe_set[value_order]
        refinement = refinement[value_order]
        # Verify safety in batches
        batch_generator = batchify((value_order, safe_set, refinement),
                                   batch_size=10000)
  
        index_to_state = self.discretization.index_to_state

        for i, (indices, safe_batch, refine_batch) in batch_generator:
           
            states = torch.tensor(index_to_state(indices), dtype=config.dtype)
            actions = self.policy(states)
            state_actions = torch.concat([states, actions], dim=1)
            next_states = self.dynamics(state_actions)
            
            decrease = self.v_decrease_bound(states, next_states)
            threshold = self.threshold(states, self.tau)
            negative = torch.squeeze(torch.less(decrease, threshold), axis=1).detach().numpy()
            safe_batch |= negative
            refine_batch[negative] = 1

            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)
            refine_bound = 0

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
              # Make sure all following points are labeled as unsafe
              safe_batch[bound:] = False
              refine_batch[bound:] = 0
              break

        # The largest index of a safe value
        max_index = i + bound + refine_bound - 1

        self.c_max = self.values[value_order[max_index]]

        # Restore the order of the safe set and adaptive refinement
        safe_nodes = value_order[safe_set]
        self.safe_set[:] = False
        self.safe_set[safe_nodes] = True
        self._refinement[value_order] = refinement

        # Ensure the initial safe set is kept
        if self.initial_safe_set is not None:
            self.safe_set[self.initial_safe_set] = True
            self._refinement[self.initial_safe_set] = 1

  
   



 



def perturb_actions(states, actions, perturbations, limits=None):
    
    num_states, state_dim = states.shape
    states_new = states.repeat_interleave(len(perturbations), dim=0)
    # generate perturbations from perturbations around baseline policy
    actions_new = actions.repeat_interleave(len(perturbations), dim=0) + perturbations.repeat(num_states,1)
    state_actions = torch.concat([states_new, actions_new], dim=1)

    if limits is not None:
        # Clip the actions
        perturbations = state_actions[:, state_dim:]
        torch.clip(perturbations, limits[:, 0], limits[:, 1], out=perturbations)
        # Remove rows that are not unique
        #Todo
        #state_actions = unique_rows(state_actions)

    return state_actions



class Dynamics():
    def __init__(self, functions):
        self.num_function = len(functions)
        self.functions = functions

    def __call__(self, state_actions):
        offset = 0
        next_states_mean = torch.tensor([],dtype=torch.float64)
        next_states_vars = torch.tensor([],dtype=torch.float64)
        for f in self.functions:
            mean, var = f.predict(state_actions)
            next_states_mean = torch.cat((next_states_mean, mean), axis=1)
            next_states_vars = torch.cat((next_states_vars, var), axis=1)
            offset += f.input_dim
    
        return next_states_mean, next_states_vars

    def add_data_point(self, state_actions, measurements):
        for i,f in enumerate(self.functions):
            f.add_data_point(state_actions, measurements[:,i].reshape(-1,1))
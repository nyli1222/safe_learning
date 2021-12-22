import cvxpy
import numpy as np
import torch

class PolicyIteration(object):

  def __init__(self, policy, dynamics, reward_function, value_function,
                 gamma=0.98):
       
    super(PolicyIteration, self).__init__()
    self.dynamics = dynamics
    self.reward_function = reward_function

    self.value_function = value_function
    self.gamma = gamma

    self.state_space = torch.tensor(self.value_function.discretization.all_points, dtype=torch.float32)
    
    #self.state_space = tf.stack(state_space, name='state_space')

    self.policy = policy
       
    

  def _run_cvx_optimization(self, next_states, rewards, **solver_options):
    # Define random variables; convert index from np.int64 to regular
    # python int to avoid strange cvxpy error; see:
    # https://github.com/cvxgrp/cvxpy/issues/380
  
    values = cvxpy.Variable(rewards.shape)

    value_matrix = self.value_function.tri.parameter_derivative(next_states)
  
    # Make cvxpy work with sparse matrices
    value_matrix = cvxpy.Constant(value_matrix)

    objective = cvxpy.Maximize(cvxpy.sum(values))
    constraints = [values <= rewards + self.gamma * value_matrix * values]
    prob = cvxpy.Problem(objective, constraints)

    # Solve optimization problem
    prob.solve(**solver_options)

    # Some error checking
    if not prob.status == cvxpy.OPTIMAL:
      raise OptimizationError('Optimization problem is {}'.format(prob.status))

    return np.array(values.value)

  def optimize_value_function(self, **solver_options):
    
    actions = self.policy(self.state_space)
    state_actions = torch.concat([self.state_space, actions],axis=1)
    next_states = self.dynamics(state_actions)
 
    # Only use the mean dynamics
    if isinstance(next_states, tuple):
      next_states, var = next_states

      rewards = self.reward_function(self.state_space,
                                       actions)

      values = self._run_cvx_optimization(next_states,
                                            rewards,
                                            **solver_options)

    
    
    self.value_function.tri.parameters = values


  def future_values(self, states, policy=None, actions=None, lyapunov=None,
                      lagrange_multiplier=1.):
   
    if actions is None:
      if policy is None:
        policy = self.policy
      actions = policy(states)

    state_actions = torch.concat([states, actions],axis=1)
    next_states = self.dynamics(state_actions)
    rewards = self.reward_function(state_actions)

    # Only use the mean dynamics
    if isinstance(next_states, tuple):
      next_states, var = next_states

    expected_values = self.value_function(next_states)

    # Perform value update
    updated_values = rewards + self.gamma * expected_values

    
    return updated_values
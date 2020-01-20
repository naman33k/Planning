import jax.numpy as np 
import jax

## This function solves a simple backward DP on a time varying LQR system like this
## min \sum_{t} Q_t(x_t,u_t) + Q_F(x_t, u_t) where x_{t+1} = Ax_t + Bu_t 
## The application on IPA algorithms is with the gradient system and deltas
def LQRSolver(cost_params, d_params, h, state_size, mu=0.0):
  c_x, c_u, c_xx, c_ux, c_uu = cost_params[h]
  ### Setup backward induction
  ### We need to define the Value function and their derivatives. 
  V_xx = c_xx
  V_x = c_x
  #print(c_x, c_xx)
  ## Now we can iterate to define the successive quadratic models
  sol_k = []
  sol_K = []
  for t in range(1, h+1):
    d_x, d_u = d_params[h-t]
    c_x, c_u, c_xx, c_ux, c_uu = cost_params[h-t]
    Q_x = c_x + d_x.T@V_x
    Q_u = c_u + d_u.T@V_x
    Q_xx = c_xx + d_x.T@V_xx@d_x
    reg = mu * np.eye(state_size)
    Q_ux = c_ux + d_u.T@(V_xx + reg)@d_x
    Q_uu = c_uu + d_u.T@(V_xx + reg)@d_u
    
    sol_k.append(-np.linalg.solve(Q_uu, Q_u))
    sol_K.append(-np.linalg.solve(Q_uu, Q_ux))
    ### Now we can update the value function
    ### These are the default updates without regularization
    # V_x = Q_x - K.T@Q_uu@k
    # V_xx = Q_xx - K.T@Q_uu@K
    ### With regularization these updates are
    # Eq (11b).
    V_x = Q_x + sol_K[t-1].T@Q_uu@sol_k[t-1] + sol_K[t-1].T@Q_u + Q_ux.T@sol_k[t-1]
    V_xx = Q_xx + sol_K[t-1].T@Q_uu@sol_K[t-1] + sol_K[t-1].T@Q_ux + Q_ux.T@sol_K[t-1]
    V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.
  return list(reversed(sol_k)), list(reversed(sol_K))

def rollout_for_actions(task, old_actions, gains, gainsK, old_pivots, alpha, mode=1):
  ## Mode = 1 - real dynamics is used to generate delta.
  ## Mode = 2 - sim dynamics is used to generate delta.
  ## Mode = 3 - to be implemented - linear dynamics is used to generate delta.
  states = [task.initial_state]
  new_actions = []
  for j in range(task.h):
    new_action = old_actions[j]+alpha*gains[j]+gainsK[j]@(states[j]-old_pivots[j])
    if mode==1:
      states.append(task.step(states[j],new_action,j, is_real_dynamics=True))
    elif mode==2:
      states.append(task.step(states[j],new_action,j, is_real_dynamics=False))
    elif mode==3:
      raise NotImplementedError
    new_actions.append(new_action)  
  return new_actions

def trajectory_cost(task, actions, is_real_dynamics=True):
  states = [task.initial_state]
  cost = 0
  for j in range(task.h):
    cost += task.cost(states[j], actions[j], j)
    states.append(task.step(states[j], actions[j], j, is_real_dynamics=is_real_dynamics))
  return cost + task.cost(states[task.h], None, task.h)  

def rollout(task, actions, is_real_dynamics=True, is_real_derivatives=False):
  states = [task.initial_state] 
  cost_params = []
  d_params = []
  total_cost = 0
  for j in range(task.h):
    total_cost+= task.cost(states[j], actions[j], j)
    cost_params.append(task.cost_grad(states[j], actions[j],j))
    d_params.append(task.dynamics_grad(states[j], actions[j],j, is_real_derivatives))
    states.append(task.step(states[j],actions[j],j, is_real_dynamics))
  total_cost+= task.cost(states[task.h], None, task.h)
  cost_params.append(task.cost_grad(states[task.h], None,task.h))
  return states, cost_params, d_params, total_cost

def IPA(mode, task, initial_actions, iters, alpha=1.0, backtracking_line_search=True, mu_min=1e-6, delta_0=2.0, mu_max=1e10):
  if mode == 'nominal':
    rollout_is_real_dynamics, rollout_is_real_derivatives = False, False
    alpha_get_actions_mode, alpha_cost_is_real_dynamics = 2, False
    final_is_real_dynamics, final_is_real_derivatives = False, False #final_is_real_derivatives is inconsequential
  elif mode == 'oracle':
    rollout_is_real_dynamics, rollout_is_real_derivatives = True, True
    alpha_get_actions_mode, alpha_cost_is_real_dynamics = 1, True
    final_is_real_dynamics, final_is_real_derivatives = True, True #final_is_real_derivatives is inconsequential
  elif mode == 'ilc_closed':
    rollout_is_real_dynamics, rollout_is_real_derivatives = True, False
    alpha_get_actions_mode, alpha_cost_is_real_dynamics = 1, True
    final_is_real_dynamics, final_is_real_derivatives = True, False #final_is_real_derivatives is inconsequential
  elif mode == 'ilc_open':
    rollout_is_real_dynamics, rollout_is_real_derivatives = True, False
    alpha_get_actions_mode, alpha_cost_is_real_dynamics = 2, True
    final_is_real_dynamics, final_is_real_derivatives = True, False #final_is_real_derivatives is inconsequential

  ## This function implements the generic loop.
  ## Perform Rollout
  mu = 1.0
  delta = delta_0
  actions = initial_actions
  alphas = 1.1**(-np.arange(10)**2)
  cost_array = []
  for i in range(iters):
    print("In iteration ", i)
    accepted = False
    states, cost_params, d_params, total_cost = rollout(task, actions, rollout_is_real_dynamics, rollout_is_real_derivatives)
    cost_array.append(total_cost)
    print("Total Cost is ",total_cost)
    sol_k, sol_K = LQRSolver(cost_params, d_params, task.h, task.state_size, mu=mu)
    if backtracking_line_search:
      #print("In backtracking line search")
      for alpha in alphas:
        #print("Trying alpha ", alpha)
        us_new = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=alpha_get_actions_mode) #REAL, no dX
        new_cost = trajectory_cost(task, us_new, is_real_dynamics=alpha_cost_is_real_dynamics) #REAL Cost
        #print("New Cost is ", new_cost)
        if new_cost < total_cost:
          # if np.abs((J_opt - J_new) / J_opt) < tol:
          #   converged = True
          total_cost = new_cost
          actions = us_new
          # Decrease regularization term.
          delta = min(1.0, delta) / delta_0
          mu *= delta
          if mu <= mu_min:
            mu = 0.0
          accepted = True
          #print("Accepting. Mu is ", mu)
          break
      if not accepted:
        # Increase regularization term.
        delta = max(1.0, delta) * delta_0
        mu = max(mu_min, mu * delta)
        #print("Nothing got accepted. Mu is now ", mu)
        if mu_max and mu >= mu_max:
          print("exceeded max regularization term")
          break
    else:
      actions = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=alpha_get_actions_mode) # not sure, let's check
  states, cost_params, d_params, total_cost = rollout(task, actions, is_real_dynamics=final_is_real_dynamics, is_real_derivatives=final_is_real_derivatives)
  return actions, cost_array, states


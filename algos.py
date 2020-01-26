import jax.numpy as np 
import jax
import timeit

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

def rollout(task, actions, is_real_dynamics=True, real_der=False):
  states = [task.initial_state] 
  cost_params = []
  d_params = []
  total_cost = 0
  for j in range(task.h):
    total_cost+= task.cost(states[j], actions[j], j)
    cost_params.append(task.cost_grad(states[j], actions[j],j))
    if real_der:
      d_params.append(task.dynamics_real_grad(states[j], actions[j],j))
    else:
      d_params.append(task.dynamics_grad(states[j], actions[j],j))
    states.append(task.step(states[j],actions[j],j, is_real_dynamics))
  total_cost+= task.cost(states[task.h], None, task.h)
  cost_params.append(task.cost_grad(states[task.h], None,task.h))
  return states, cost_params, d_params, total_cost



def IPA(task, initial_actions, iters, alg="ILQR-FULL", alpha=1.0, backtracking_line_search=True, mu_min=1e-6, delta_0=2.0, mu_max=1e10):
  ### Modes I am implementing
  ### ILQR-FULL - This is the standard "full information" ILQR
  ### ILQR-CE - This is the closed loop certainty equivalent ILQR System 
  ### ILC-CLOSED - This is the closed loop version of ILC
  ### ILC-OPEN - This is the open loop version of ILC

  ## This function implements the generic loop.
  ## Perform Rollout
  mu = 1.0
  delta = delta_0
  actions = initial_actions
  alphas = (1.1**(-np.arange(10)**2))
  cost_array = []
  dim_x = task.state_size
  dim_u = task.action_size
  IGPC_h = 3
  if alg=="IGPC":
    M = [np.zeros((dim_u, dim_x)) for i in range(IGPC_h)]
  for i in range(iters):
    print("In iteration ", i)
    accepted = False
    if alg=="ILQR-CE":
      states, cost_params, d_params, total_cost = rollout(task, actions, is_real_dynamics=False,real_der=False)
    elif alg=="ILC-CLOSED" or alg=="ILC-OPEN" or alg=="IGPC":
      states, cost_params, d_params, total_cost = rollout(task, actions, is_real_dynamics=True,real_der=False)
    elif alg=="ILQR-FULL":
      states, cost_params, d_params, total_cost = rollout(task, actions, is_real_dynamics=True,real_der=True)
    else:
      print("Algorithm not recognized")
      exit()
    cost_array.append(total_cost)
    print("Total Cost is ",total_cost)
    sol_k, sol_K = LQRSolver(cost_params, d_params, task.h, task.state_size, mu=mu)
    if backtracking_line_search:
      #print("In backtracking line search")
      for alpha in alphas:
        #print("Trying alpha ", alpha)
        if alg=="ILQR-CE":
          us_new = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=2)
          new_cost = trajectory_cost(task, us_new, is_real_dynamics=False)
        elif alg=="ILC-OPEN":
          us_new = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=2)
          new_cost = trajectory_cost(task, us_new, is_real_dynamics=True)
        elif alg=="ILC-CLOSED" or alg=="ILQR-FULL":
          us_new = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=1)
          new_cost = trajectory_cost(task, us_new, is_real_dynamics=True)
        elif alg=="IGPC":
          us_new, M_new = IGPC_rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, M, dim_x, dim_u)
          new_cost = trajectory_cost(task, us_new, is_real_dynamics=True)
        #print("New Cost is ", new_cost)
        if new_cost < total_cost:
          # if np.abs((J_opt - J_new) / J_opt) < tol:
          #   converged = True
          total_cost = new_cost
          actions = us_new
          if alg=="IGPC":
            M = M_new
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
      if alg=="ILQR-CE":
          actions = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=2)
      elif alg=="ILC-OPEN":
          actions = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=2)
      elif alg=="ILC-CLOSED" or alg=="ILQR-FULL":
          actions = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=1)
      elif alg=="IGPC":
          actions, M = IGPC_rollout_for_actions(task, actions, sol_k, sol_K, states, M, dim_x, dim_u)

  if alg=="ILQR-CE":
    ### In this we need to one real rollout for actions at the end
    print("Executing final run for ILQR-CE")
    states, cost_params, d_params, total_cost = rollout(task, actions, is_real_dynamics=False,real_der=False)
    sol_k, sol_K = LQRSolver(cost_params, d_params, task.h, task.state_size, mu=mu)
    us_new = rollout_for_actions(task, actions, sol_k, sol_K, states, alpha, mode=1)
    new_cost = trajectory_cost(task, us_new, is_real_dynamics=True)
    cost_array.append(new_cost)
    print("Final Execution is ", new_cost)
    actions = us_new
  states, cost_params, d_params, total_cost = rollout(task, actions, is_real_dynamics=True,real_der=False)

  return actions, cost_array, states

def padded_w(w, l, d):
  if l==0:
    return []
  if len(w) < l:
    prep = [np.zeros(d)]*(l-len(w))
    return prep + w
  else:
    return w[-l:]

def IGPC_rollout_for_actions(task, old_actions, gains, gainsK, old_pivots, ILC_alpha, M, dim_x, dim_u):
  IGPC_alpha = 0.01
  IGPC_delta = 1.0
  IGPC_h = len(M)
  states = [task.initial_state]
  new_actions = []
  ILC_actions = []
  w = []
  for j in range(task.h):
    new_w = states[j]-old_pivots[j]
    w.append(new_w)
    new_ilc_action = old_actions[j]+ILC_alpha*gains[j]+gainsK[j]@new_w
    ILC_actions.append(new_ilc_action)
    if j > IGPC_h:
      w_to_pass = padded_w(w, IGPC_h+len(M), dim_x)
      #start_time = timeit.default_timer()
      grad = IGPC_Gradient(task, M, w_to_pass, ILC_actions[-IGPC_h:], states[-(IGPC_h+1)], IGPC_h, dim_x, dim_u)
      #elapsed = timeit.default_timer() - start_time
      #print("Elapsed ", elapsed)
      ### update M
      M = [M[i] - IGPC_alpha*grad[i] for i in range(IGPC_h)]
      #print(len(M))
      #print(M[0])
      ### Compute IGPC delta Action
      IGPC_act = sum([M[i]@w_to_pass[-i] for i in range(IGPC_h)])
      new_action = new_ilc_action+IGPC_delta*IGPC_act  
      states.append(task.step(states[j],new_action,j, is_real_dynamics=True))
      new_actions.append(new_action)
    else:
      new_action = new_ilc_action  
      states.append(task.step(states[j],new_action,j, is_real_dynamics=True))
      new_actions.append(new_action)  
  return new_actions, M  

def IGPC_Gradient(task, M, w, actions, initial_state, IGPC_h, dim_x, dim_u):
  hm = len(M)
  ## The length of w should be the sum of hm and IGPC_h
  ## The first hm entries of w represent the "previous" perturbations
  start_index = hm
  index = start_index
  ## Simple check assertions to debug for errors
  for i in range(hm):
    assert(M[i].shape==(dim_u, dim_x))
  assert(len(w)==hm+IGPC_h)

  ## We want to create a list of 2d values representing gradient
  cost_der = [np.zeros((dim_u, dim_x)) for i in range(hm)]
  ## Initialize x and der_x
  x = initial_state
  der_x = [np.zeros((dim_x, dim_u, dim_x)) for i in range(hm)]

  ## --------- LOGIC -----------------------
  ## Forward Pass
  ## a_t = u_t + \sum_{0 to hm} M_i w_{t-1-i}
  ## x_{t+1} = g(x_t, a_t) + w_t
  ## del x_{t+1} / del M_i = first_term + second_term
  ## first_term = del g(x_t, a_t)/del x * del x_t / del M_i
  ## * is an einsum defined as 'ij,jkl->ikl'
  ## second_term = del g(x_t, a_t)/ del u * del a_t / del M_i
  ## del a_t / del M_i -> array with only entries at [j,j,k] = [w_t-1-i]_k

  ## We can compute the cost derivative pass here itself
  ## del cost(x_t,a_t)/ del M_i = del cost/ del x * del x/del M_i + del cost/ del u * del u/del M_i
  ## * is the einsum 'i,ijk->jk'
  ## ----------------------------------------
  for t in range(IGPC_h):
    ## Compute a_t
    a = actions[t]
    for i in range(hm):
      a += M[i]@w[start_index+t-1-i]
    next_x = task.step(x, a, 0, is_real_dynamics=False) + w[start_index+t]
    ## Computing derivatives of next_x
    der_x_next = []
    for i in range(hm):
      d_x, d_u = task.dynamics_grad(x, a, 0)
      ## Perform the einsum
      ### CHECK
      #first_term = np.einsum('ij,jkl->ikl', d_x, der_x[i])
      first_term = np.tensordot(d_x, der_x[i],1)
      ## Prepare der_a
      der_a = np.zeros((dim_u, dim_u, dim_x))
      for j in range(dim_u):
        der_a = jax.ops.index_update(der_a, jax.ops.index[j,j,:], w[start_index+t-1-i])  
      ## Perform the einsum
      #second_term = np.einsum('ij,jkl->ikl', d_u, der_a)
      second_term = np.tensordot(d_u, der_a,1)
      der_x_next.append(first_term+second_term)

      ## Compute cost derivatives
      c_x, c_u, c_xx, c_ux, c_uu = task.cost_grad(x, a, 0)
      #cost_der_1 = np.einsum('i,ijk->jk', c_x, der_x[i])
      cost_der_1 = np.tensordot(c_x, der_x[i],1)
      #cost_der_2 = np.einsum('i,ijk->jk', c_u, der_a)
      cost_der_2 = np.tensordot(c_u, der_a,1)
      cost_der[i] = cost_der[i] + cost_der_1 + cost_der_2
    x = next_x
    der_x = der_x_next
  return cost_der



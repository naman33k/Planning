import jax
import jax.numpy as np 
from algos import *

def counterfact_loss(E, off, W, H, M, t, task, old_actions, gains, gainsK, old_pivots, state):
    y, cost = state, 0
    for h in range(H):
        u_extra = np.tensordot(E, W[h:h+M], axes = ([0, 2], [0, 1])) + off
        new_action = old_actions[t+h]+gains[t+h]+gainsK[t+h]@(y-old_pivots[t+h])+u_extra
        cost = task.cost(y, new_action, t+h)
        y = task.step(y, new_action, t+h, is_real_dynamics = False) + W[h+M]
    return cost

grad_loss = jax.grad(counterfact_loss, (0,1))

def moo(task, old_actions, gains, gainsK, old_pivots, is_real_dynamics=True, is_real_derivatives=False):
    H, M, lr = 3, 3, 1e-5
    states, new_actions, cost_params, d_params, cost = [task.initial_state], [], [], [], 0
    E, W, off = np.zeros((H, task.action_size, task.state_size)), np.zeros((H+M, task.state_size)), np.zeros(task.action_size)
    for j in range(task.h):
        u_extra = np.tensordot(E, W[-M:], axes = ([0, 2], [0, 1])) + off

        new_action = old_actions[j]+gains[j]+gainsK[j]@(states[j]-old_pivots[j])+u_extra
        new_actions.append(new_action)  
        states.append(task.step(states[j],new_action,j, is_real_dynamics))
        cost += task.cost(states[j], new_action, j)
        cost_params.append(task.cost_grad(states[j], new_action,j))
        d_params.append(task.dynamics_grad(states[j], new_action, j, is_real_derivatives))

        W = jax.ops.index_update(W, 0, states[j+1] - task.step(states[j], new_action, j, is_real_dynamics = False))
        W = np.roll(W, -1, axis = 0)
        if j >= H:
            delta_E, delta_off = grad_loss(E, off, W, H, M, j-H, task, old_actions, gains, gainsK, old_pivots, states[j-H])
            E -= 0 * delta_E
            off -= lr * delta_off
            if j % 20 == 0:
                print(off)
    cost += task.cost(states[task.h], None, task.h)
    cost_params.append(task.cost_grad(states[task.h], None,task.h))
    return states, new_actions, cost_params, d_params, cost

def PiLOT(task, initial_actions, iters):
    actions, cost = initial_actions, 0
    gains = [np.zeros(task.action_size) for _ in range(task.h)]
    old_pivots = [np.zeros(task.state_size) for _ in range(task.h)]
    gainsK = [np.zeros((task.action_size, task.state_size)) for _ in range(task.h)]
    for _ in range(iters):
       old_pivots, actions, cost_params, d_params, cost = moo(task, actions, gains, gainsK, old_pivots)
       gains, gainsK = LQRSolver(cost_params, d_params, task.h, task.state_size, mu=1.)
       print("Iter %d Cost %f"%(_, cost))
    states, actions, cost_params, d_params, cost = moo(task, actions, gains, gainsK, old_pivots)
    return actions, None, states, cost
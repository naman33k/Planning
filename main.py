import jax
import time
import jax.numpy as np
from algos import *
from tasks import *

key = jax.random.PRNGKey(10)

# action_size = 1
# pendulum = PendulumTask()
# u0 = jax.random.uniform(key, shape=(pendulum.h, action_size), minval=-1.0)
# final_actions, costs, states = IPA(pendulum, u0, 100)
# print(costs)
# print("Rendering the found action now")
# for i in range(pendulum.h+1):
# 	if i==0:
# 		pendulum.render(states[i])
# 		time.sleep(1. / 30.)
# 	else:
# 		pendulum.render(states[i], last_u=final_actions[i-1])
# 		time.sleep(1. / 30.)

action_size = 2
quadrotor = PlanarQuadrotor()

u0 = np.tile(quadrotor.hover_input, (quadrotor.h, 1)) + 0.0*jax.random.uniform(key, shape=(quadrotor.h, quadrotor.action_size), minval=-1.0)
# print(quadrotor.initial_state)
# print(quadrotor.step(quadrotor.initial_state, quadrotor.hover_input, 1, is_real_dynamics=False))

final_actions, costs, states, final_cost = IPA('ilc_closed', quadrotor, u0, 6)
print('Final Costs %f'%final_cost)
for i in range(quadrotor.h+1):
	if i==0:
		quadrotor.render(states[i])
		time.sleep(1. / 20.)
	else:
		quadrotor.render(states[i], last_u=final_actions[i-1])
		time.sleep(1. / 20.)

# Wind = 0.1 (10 iters), nominal = 52, oracle = 60, ilc_closed = 60, ilqr_closed = 90, ilc_open = 0.2M, ilqr_open = 0.7M
# Wind = 0.1 (30 iters), nominal = 52, oracle = 60, ilc_closed = 60, ilqr_closed = 0.7M (K->0?), ilc_open = 0.2M, ilqr_open = 0.7M
# Wind = 0.15 (10 iters), nominal = 52, oracle = 65, ilc_closed = 25K, ilqr_closed = 260, ilc_open = 1M, ilqr_open = 8M
# Wind = 0.15 (30 iters), nominal = 52, oracle = 65, ilc_closed = 65, ilqr_closed = 8M (K->0?), ilc_open = 1M, ilqr_open = 8M
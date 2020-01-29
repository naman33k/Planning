import jax
import time
import jax.numpy as np
from algos import *
from tasks import *
import matplotlib.pyplot as plt

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

key = jax.random.PRNGKey(10)
action_size = 2
quadrotor = PlanarQuadrotor()
u0 = np.tile(quadrotor.hover_input, (quadrotor.h, 1)) + 0.0*jax.random.uniform(key, shape=(quadrotor.h, quadrotor.action_size), minval=-1.0)

final_actions, costs, states, rollout_vals = IPA(quadrotor, u0, 2, alg="ILC-CLOSED")
print(rollout_vals)
print([float(c) for c in costs])
print(states[-1])
print("Rendering the found action now")
# devs = [quadrotor.step(states[i], final_actions[i],1)[3] - quadrotor.step(states[i], final_actions[i], 1, is_real_dynamics=False)[3] for i in range(quadrotor.h)]
# print(devs)
# plt.plot(devs)
# plt.show()
for i in range(quadrotor.h+1):
	if i==0:
		quadrotor.render(states[i])
		time.sleep(1. / 20.)
	else:
		quadrotor.render(states[i], last_u=final_actions[i-1])
		time.sleep(1. / 20.)



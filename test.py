from tasks import *
import time
import numpy as np
import jax
from algos import *
# env = PlanarQuadrotor()
# for t in np.linspace(0., 1., num=100, endpoint=True):
#     env.render(np.array([t, t**2, np.pi*t*1., 0., 0., 0.]))
#     time.sleep(1)

key = jax.random.PRNGKey(10)
action_size = 1
pendulum = PendulumTask()
u0 = jax.random.uniform(key, shape=(pendulum.h, action_size), minval=-1.0)
final_actions, costs, states = IPA(pendulum, u0, 0)
print(costs)
print("Rendering the found action now")
for i in range(pendulum.h+1):
	if i==0:
		pendulum.render(states[i])
		time.sleep(1. / 30.)
	else:
		pendulum.render(states[i], last_u=final_actions[i-1])
		time.sleep(1. / 30.)

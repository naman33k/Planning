import jax
import time
import jax.numpy as np
from algos import *
from tasks import *

key = jax.random.PRNGKey(10)
action_size = 1
pendulum = PendulumTask()
u0 = jax.random.uniform(key, shape=(pendulum.h, action_size), minval=-1.0)
final_actions, costs, states = IPA(pendulum, u0, 100)
print(costs)
print("Rendering the found action now")
for i in range(pendulum.h+1):
	if i==0:
		pendulum.render(states[i])
		time.sleep(1. / 30.)
	else:
		pendulum.render(states[i], last_u=final_actions[i-1])
		time.sleep(1. / 30.)
import jax
import jax.numpy as np
from algos import *
from tasks import *

key = jax.random.PRNGKey(10)
action_size = 1
pendulum = PendulumTask()
u0 = jax.random.uniform(key, shape=(pendulum.h, action_size), minval=-1.0)
final_actions, costs, states = IPA(pendulum, u0, 200)
print(costs)
print(states)
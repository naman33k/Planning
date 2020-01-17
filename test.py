from tasks import *
import time
import numpy as np

env = PlanarQuadrotor()
for t in np.linspace(0., 1., num=100, endpoint=True):
    env.render(np.array([t, t**2, np.pi*t*1., 0., 0., 0.]))
    time.sleep(0.01)
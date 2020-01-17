from tasks import *
import time
import numpy as np

env = PlanarQuadrotor()
env.render(np.array([1., 1., np.pi/6, 0., 0., 0.]))
time.sleep(100)
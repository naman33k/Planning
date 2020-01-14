import jax
import jax.numpy as np

class PendulumTask:
  def __init__(self):
    self.min_bounds=-1.0
    self.max_bounds=1.0
    self.m=1.0
    self.l=1.0
    self.g=9.80665
    self.state_size = 3
    self.action_size = 1
    self.pendulum_length = 1.0
    self.x_goal = self.augment_state(np.array([0.0, 0.0]))
    self.initial_state = self.augment_state(np.array([np.pi, 0.0]))
    self.h = 300
    self.dt = 0.02

    @jax.jit
    def _dynamics(input_val):
      diff = (self.max_bounds - self.min_bounds) / 2.0
      mean = (self.max_bounds + self.min_bounds) / 2.0
      x, u = input_val
      u = diff * np.tanh(u) + mean
      sin_theta = x[0]
      cos_theta = x[1]
      theta_dot = x[2]
      torque = u[0]
      theta = np.arctan2(sin_theta, cos_theta)
      theta_dot_dot = -3.0*self.g/(2*self.l)*np.sin(theta+np.pi)
      theta_dot_dot += 3.0 / (self.m * self.l**2) * torque
      next_theta = theta + theta_dot * self.dt
      return np.array([np.sin(next_theta), np.cos(next_theta), theta_dot + theta_dot_dot * self.dt])

    self._dynamics = _dynamics
    self._dynamics_der = jax.jit(jax.jacfwd(_dynamics))

    self.Q = np.eye(self.state_size)
    self.Q = jax.ops.index_update(self.Q,(0, 1),self.pendulum_length)
    self.Q = jax.ops.index_update(self.Q,(1, 0),self.pendulum_length)
    self.Q = jax.ops.index_update(self.Q,(0, 0),self.pendulum_length**2)
    self.Q = jax.ops.index_update(self.Q,(1, 1),self.pendulum_length**2)
    self.Q = jax.ops.index_update(self.Q,(2, 2),0.0)
    self.Q_terminal = 100 * np.eye(self.state_size)
    self.R = np.array([[0.1]])

    def _costval(x, u, i):
      if i == self.h:
        return (x-self.x_goal).T@self.Q_terminal@(x-self.x_goal)
      else:
        return (x-self.x_goal).T@self.Q@(x-self.x_goal) + u.T@self.R@u

    def _costgrad(x,u,i):
      if i==self.h:
        return [2*self.Q_terminal@(x - self.x_goal), np.zeros((1,)), 2*self.Q_terminal, np.zeros((self.action_size, self.state_size)), np.zeros((self.action_size, self.action_size))]
      else:
        return [2*self.Q@(x-self.x_goal), 2*self.R@u, 2*self.Q, np.zeros((self.action_size,self.state_size)), 2*self.R]
    
    self._cost = _costval
    self._costgrad = _costgrad

  def step(self,x,u,i):
    return self._dynamics([x,u])

  def dynamics_grad(self,x,u,i):
    return self._dynamics_der([x,u])

  def cost_grad(self,x,u,i):
    return self._costgrad(x,u,i)

  def cost(self,x,u,i):
    return self._cost(x,u,i)

  def augment_state(self,x):
    return np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

  def compress_state(self, x):
    return np.array([np.arctan2(x[0], x[1]), x[2]])
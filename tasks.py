import pyglet
import jax
import jax.numpy as np
import os
# necessary for rendering
from gym.envs.classic_control import rendering

class PendulumTask:
  metadata = {'render.modes' : ['human', 'rgb_array'],'video.frames_per_second' : 30}
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
    self.viewer = None

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

    def _dynamics_real(input_val):
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
      theta_dot_dot += 3.0 / (1.1 * 1.1**2) * torque
      next_theta = theta + theta_dot * self.dt
      return np.array([np.sin(next_theta), np.cos(next_theta), theta_dot + theta_dot_dot * self.dt])


    self._dynamics = _dynamics
    self._dynamics_real = _dynamics_real
    self._dynamics_der = jax.jit(jax.jacfwd(_dynamics))

    self.Q = np.eye(self.state_size)
    # self.Q = jax.ops.index_update(self.Q,(0, 1),self.pendulum_length)
    # self.Q = jax.ops.index_update(self.Q,(1, 0),self.pendulum_length)
    # self.Q = jax.ops.index_update(self.Q,(0, 0),self.pendulum_length**2)
    # self.Q = jax.ops.index_update(self.Q,(1, 1),self.pendulum_length**2)
    self.Q = jax.ops.index_update(self.Q,(2, 2),0.0)
    self.Q_terminal = 100 * np.eye(self.state_size)
    self.R = np.array([[0.1]])

    def _costval(x, u, i):
      #print("asa")
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

  def step(self,x,u,i, is_real_dynamics=True):
    if is_real_dynamics:
      return self._dynamics_real([x,u])
    else:
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

  def render(self, state, mode='human', last_u=None):
    if self.viewer is None:
        self.viewer = rendering.Viewer(500,500)
        self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
        rod = rendering.make_capsule(1, .2)
        rod.set_color(.8, .3, .3)
        self.pole_transform = rendering.Transform()
        rod.add_attr(self.pole_transform)
        self.viewer.add_geom(rod)
        axle = rendering.make_circle(.05)
        axle.set_color(0,0,0)
        self.viewer.add_geom(axle)
        fname = "clockwise.png"
        self.img = rendering.Image(fname, 1., 1.)
        self.imgtrans = rendering.Transform()
        self.img.add_attr(self.imgtrans)

    self.viewer.add_onetime(self.img)
    self.pole_transform.set_rotation(np.arctan2(state[0],state[1]) + np.pi/2)
    if last_u:
        self.imgtrans.scale = (last_u/2, np.abs(last_u)/2)
    return self.viewer.render(return_rgb_array = mode=='rgb_array')

  def close(self):
    if self.viewer:
        self.viewer.close()
        self.viewer = None 


class PlanarQuadrotor:
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=9.81):
        self.initialized = False
        self.dt=.05
        self.m = 0.1 # kg
        self.L = 0.2 # m
        self.I = 0.004 # inertia, kg*m^2
        self.g = g
        self.hover_input = np.array([self.m*self.g/2., self.m*self.g/2.])
        self.viewer = None
        self.action_space = (2,)
        self.wind_force = 1.
        self.n = 6
        self.observation_space = (self.n,)
        self.initial_state = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        @jax.jit
        def _dynamics(x, u):
            state = x
            x, y, th, xdot, ydot, thdot = state
            u1, u2 = u
            g = self.g
            m = self.m 
            L = self.L
            I = self.I
            dt = self.dt
            xddot = -(u1+u2)*np.sin(th)/m # xddot
            yddot = (u1+u2)*np.cos(th)/m - g # yddot
            thddot = L*(u2 - u1)/I # thetaddot
            state_dot = np.array([xdot, ydot, thdot, xddot, yddot, thddot])
            new_state = state + state_dot*dt
            return new_state

        @jax.jit
        def _wind_field(x,y):
            return [self.wind_force*x, self.wind_force*y]
        self.wind_field = _wind_field
        
        @jax.jit
        def _dynamics_real(x, u):
            state = x
            x, y, th, xdot, ydot, thdot = state
            u1, u2 = u
            g = self.g
            m = self.m 
            L = self.L
            I = self.I
            dt = self.dt
            wind = self._wind_field(x,y)
            xddot = -(u1+u2)*np.sin(th)/m + wind[0]/m # xddot
            yddot = (u1+u2)*np.cos(th)/m - g + wind[1]/m # yddot
            thddot = L*(u2 - u1)/I # thetaddot
            state_dot = np.array([xdot, ydot, thdot, xddot, yddot, thddot])
            new_state = state + state_dot*dt
            return new_state
        self._dynamics = _dynamics

        def _costval(x, u, i):
            if i==task.h:
              return 100*np.linalg.norm(x - self.goal_state)**2
            else:
              return np.linalg.norm(x - self.goal_state)**2 + 0.1*np.linalg.norm(u - self.hover_input)**2
      
        def _costgrad(x,u,i):
          if i==self.h:
            return [2*self.Q_terminal@(x - self.x_goal), np.zeros((1,)), 2*self.Q_terminal, np.zeros((self.action_size, self.state_size)), np.zeros((self.action_size, self.action_size))]
          else:
            return [2*self.Q@(x-self.x_goal), 2*self.R@u, 2*self.Q, np.zeros((self.action_size,self.state_size)), 2*self.R]
        
        self._cost = _costval
        self._costgrad = _costgrad

    def step(self,u):
        self.last_u = u
        state = self._dynamics(self.state, u)
        self.state = state
        return self.state

    def linearize_dynamics(self, x0, u0):
        # Linearize dynamics about x0, u0
        dyn_jacobian = jax.jit(jax.jacrev(self._dynamics, argnums=(0,1))) 
        F = dyn_jacobian(x0, u0)
        A = F[0]
        B = F[1]
        # F = np.hstack(dyn_jacobian(x0, u0)) 
        return A, B

    def reset(self):
        x = random.uniform(generate_key(), minval=-0.5, maxval=0.5)
        y = random.uniform(generate_key(), minval=-0.5, maxval=0.5)
        th = random.uniform(generate_key(), minval=-30*np.pi/180, maxval=30*np.pi/180)
        xdot = random.uniform(generate_key(), minval=-0.1, maxval=0.1)
        ydot = random.uniform(generate_key(), minval=-0.1, maxval=0.1)
        thdot = random.uniform(generate_key(), minval=-0.1, maxval=0.1)
        
        self.state = np.array([x, y, th, xdot, ydot, thdot])
        self.last_u = np.array([0.0, 0.0])
        return self.state 

    def render(self, state, mode='human', last_u=None):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000,1000)
            self.viewer.set_bounds(-0.2, 1.2, -0.2, 1.2)
            fname = "drone3.png"
            self.img = rendering.Image(fname, 0.4, 0.17)
            self.img.set_color(1., 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            fnamewind = "wind.png"
            self.imgwind = rendering.Image(fnamewind, 1.4, 1.4)
            self.imgwind.set_color(0.3, 0.3, 0.3)
            self.imgtranswind = rendering.Transform()
            self.imgwind.add_attr(self.imgtranswind)

        self.viewer.add_onetime(self.imgwind)
        self.viewer.add_onetime(self.img)
        self.imgtrans.set_translation(state[0], state[1]+0.04)
        self.imgtrans.set_rotation(state[2])
        self.imgtranswind.set_translation(0.5, 0.5)

        #if last_u:
        #    self.imgtrans.scale = (last_u/2, np.abs(last_u)/2)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None 

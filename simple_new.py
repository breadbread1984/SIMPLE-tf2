#!/usr/bin/python3
import numpy as np;
import tensorflow as tf;

class Simple(object):
  def __init__(self, nx = 40, ny = 40, dtype = tf.float32):
    self.dtype = dtype;
    self.nx = nx;
    self.ny = ny;
    # domain discretization
    self.x, self.y, self.dx, self.dy = self.domain_discretization();
    # fluid properties
    self.rho = self.fluid_properties();
    # velocities, pressure initialization
    self.u, self.v = self.initialization();
    # create graphs
    self.circ_v = tf.Graph();
    with self.circ_v.as_default():
      self.x_pos, self.y_pos, self.x_u, self.x_v = self.CirV();
    self.cdiff = tf.Graph();
  def domain_discretization(self,):
    dx = 1/self.nx;
    dy = 1/self.ny;
    x = [i * dx for i in range(self.nx)];
    y = [i * dy for i in range(self.ny)];
    x = np.tile(np.reshape(x, (-1,1,1)), (1, self.ny)); # x.shape = (nx, ny, nz)
    y = np.tile(np.reshape(y, (1,-1,1)), (self.nx, 1)); # y.shape = (nx, ny, nz)
    return x, y, dx, dy;
  def fluid_properties(self,):
    rho = 1.; # density
  def initialization(self,):
    u = 0.1 * np.ones_like(self.x);
    v = 0.1 * np.ones_like(self.y);
    return u, v;
  def CircV(self,):
    x_pos = tf.placeholder((self.nx, self.ny), dtype = tf.float32);
    y_pos = tf.placeholder((self.nx, self.ny), dtype = tf.float32);
    r = ((x_pos - 1. / 2) ** 2 + (y_pos - 1 / 2) ** 2) ** 0.5;
    denominat_temp = y_pos - 1 / 2;
    numerator_temp = x_pos - 1 / 2;
    theta = tf.math.atan2(denominat_temp, numerator_temp);
    x_u = -r * tf.math.sin(theta);
    x_v = r * tf.math.cos(theta);
    return x_pos, y_pos, x_u, x_v;
  def CDiff(self,):
    
  def solve(self,):
    with tf.Session(graph = self.circ_v) as sess:
      self.u, self.v = sess.run([self.x_u.outputs[0], self.x_v.outputs[0]],
                                feed_dict = {self.x_pos.outputs[0]: self.x,
                                             self.y_pos.outputs[0]: self.y});
    

#!/usr/bin/python3

from math import pi;
import tensorflow as tf;

class SIMPLE(object):
  def __init__(self, nx=20,ny=30,nz=20):
    self.nx = nx;
    self.ny = ny;
    self.nz = nz;
    # domain discretization
    self.x, self.y, self.z = self.domain_discretization();
    # fluid properties
    self.rho, self.mu = self.fluid_properties();
    # underelaxation properties
    self.omega_u, self.omega_v, self.omega_w, self.omega_p, self.omega_pp, self.beta = self.underelaxation_properties();
    # velocities, pressure initialization
    self.u, self.v, self.w, self.P = self.initialization();
    # set conditions 
    self.setConditions();
  def domain_discretization(self,):
    x = tf.constant([i * 1/self.nx for i in range(self.nx + 1)]); # x in range [0,1] quantized into nx+1 values
    y = tf.constant([i * 2*pi/(self.ny + 1) for i in range(self.ny + 1)]); # y in range [0, 2 * pi] quantized into ny+1 values
    z = tf.constant([i * 1/self.nz for i in range(self.nz + 1)]); # z in range [0,1] quantized into nz+1 values
    # NOTE: x.shape = (nx+1,) y.shape = (ny + 1) z.shape = (nz + 1)
    return x,y,z;
  def fluid_properties(self,):
    # NOTE: derivable member function
    rho = tf.constant(1);
    mu = tf.constant(0.01);
    return rho, mu;
  def underelaxation_properties(self,):
    # NOTE: derivable member function
    omega_u = tf.constant(.5);
    omega_v = tf.constant(.5);
    omega_w = tf.constant(.5);
    omega_p = tf.constant(.1);
    omega_pp = tf.constant(1.7);
    beta = tf.constant(0.95);
    return omega_u, omega_v, omega_w, omega_p, omega_pp, beta;
  def initialization(self,):
    # NOTE: derivable member function
    indices_x = tf.tile(tf.reshape(tf.range(1,self.nx), (-1,1,1)), (1,self.ny-1,self.nz-1)); # xx.shape = (nx-1,ny-1,nz-1)
    indices_y = tf.tile(tf.reshape(tf.range(1,self.ny), (1,-1,1)), (self.nx-1,1,self.nz-1)); # yy.shape = (nx-1,ny-1,nz-1)
    indices_z = tf.tile(tf.reshape(tf.range(1,self.nz), (1,1,-1)), (self.nx-1,self.ny-1,1)); # zz.shape = (nx-1,ny-1,nz-1)
    indices = tf.stack([indices_x,indices_y,indices_z], axis = -1); # indices.shape = (nx-1,ny-1,nz-1,3)
    u = tf.zeros((self.nx+1, self.ny+1, self.nz+1)); # u.shape = (21, 31, 21)
    v = tf.scatter_nd(indices = indices, updates = tf.ones((self.nx-1,self.ny-1,self.nz-1)), shape = (self.nx+1,self.ny+1,self.nz+1)); # v.shape = (21, 31, 21)
    w = tf.zeros((self.nx+1, self.ny+1, self.nz+1)); # w.shape = (21, 31, 21)
    P = tf.zeros((self.nx+1, self.ny+1, self.nz+1)); # P.shape = (21, 31, 21)
    return u,v,w,P;
  def setConditions(self,):
    # NOTE: derivable member function
    omega = constant(1/4);
    indices_x = tf.reshape(tf.math.logical_and(tf.math.less_equal(self.x, 0.4), tf.math.not_equal(self.x, 0)), (-1, 1, 1)); # indices_x.shape = (nx + 1, 1, 1)
    indices_y = tf.reshape(tf.cast(tf.ones_like(self.y), dtype = tf.bool), (1, -1, 1)); # indices_y.shape = (1, ny + 1, 1)
    indices_z = tf.reshape(tf.math.logical_and(tf.math.greater_equal(self.z, 0.2), tf.math.not_equal(self.z, 0)), (1, 1, -1,)); # indices_z.shape = (1, 1, nz + 1)
    mask_x = tf.tile(indices_x, (1, self.ny + 1, self.nz + 1)); # mask_x.shape = (nx + 1, ny + 1, nz + 1)
    mask_y = tf.tile(indices_y, (self.nx + 1, 1, self.nz + 1)); # mask_y.shape = (nx + 1, ny + 1, nz + 1)
    mask_z = tf.tile(indices_z, (self.nx + 1, self.ny + 1, 1)); # mask_z.shape = (nx + 1, ny + 1, nz + 1)
    mask = tf.math.logical_and(tf.math.logical_and(mask_x, mask_y), mask_z); # mask.shape = (nx + 1, ny + 1, nz + 1)

if __name__ == "__main__":
  simple = SIMPLE();

#!/usr/bin/python3

from math import pi;
import tensorflow as tf;

class SIMPLE(object):
  def __init__(self, nx=20,ny=30,nz=20):
    self.nx = nx;
    self.ny = ny;
    self.nz = nz;
    # domain discretization
    self.x, self.y, self.z, self.dx, self.dy, self.dz = self.domain_discretization();
    # fluid properties
    self.rho, self.mu = self.fluid_properties();
    # underelaxation properties
    self.omega_u, self.omega_v, self.omega_w, self.omega_p, self.omega_pp, self.beta = self.underelaxation_properties();
    # velocities, pressure initialization
    self.u, self.v, self.w, self.P = self.initialization();
    # set conditions 
    self.setConditions();
  def domain_discretization(self,):
    dx = 1/self.nx;
    x = tf.constant([i * dx for i in range(self.nx + 1)]); # x in range [0,1] quantized into nx+1 values
    dy = 2*pi/(self.ny + 1);
    y = tf.constant([i * dy for i in range(self.ny + 1)]); # y in range [0, 2 * pi] quantized into ny+1 values
    dz = 1/self.nz;
    z = tf.constant([i * dz for i in range(self.nz + 1)]); # z in range [0,1] quantized into nz+1 values
    # NOTE: x.shape = (nx+1,) y.shape = (ny + 1) z.shape = (nz + 1)
    return x,y,z, dx,dy,dz;
  def fluid_properties(self,):
    # NOTE: derivable member function
    rho = 1.;
    mu = 0.01;
    return rho, mu;
  def underelaxation_properties(self,):
    # NOTE: derivable member function
    omega_u = .5;
    omega_v = .5;
    omega_w = .5;
    omega_p = .1;
    omega_pp = 1.7;
    beta = 0.95;
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
    mask_x = tf.tile(tf.reshape(tf.math.logical_and(tf.math.less_equal(self.x, 0.4), tf.math.not_equal(self.x, 0)), (-1, 1, 1)), (1, self.ny + 1, self.nz + 1)); # mask_x.shape = (nx + 1, 1, 1)
    mask_z = tf.tile(tf.reshape(tf.math.greater_equal(self.z, 0.2), (1, 1, -1,)), (self.nx + 1, self.ny + 1, 1)); # mask_z.shape = (1, 1, nz + 1)
    mask = tf.math.logical_and(mask_x, mask_z); # mask.shape = (nx + 1, ny + 1, nz + 1)
    x_value = tf.tile(tf.reshape(self.x, (-1,1,1)), (1, self.ny + 1, self.nz + 1)); # x_value.shape = (nx + 1, ny + 1, nz + 1)
    self.u = tf.where(mask, tf.zeros_like(self.u), self.u);
    self.v = tf.where(mask, 0.25 * x_value, self.v);
    self.w = tf.where(mask, tf.zeros_like(self.w), self.w);
    self.u = tf.concat([self.u[:,-2:-1,:], self.u[:,-2:-1,:], self.u[:,2:-1,:], self.u[:,-2:-1,:]], axis = 1);
    self.v = tf.concat([self.v[:,-2:-1,:], self.v[:,-2:-1,:], self.v[:,2:-1,:], self.v[:,-2:-1,:]], axis = 1);
    self.w = tf.concat([self.w[:,-2:-1,:], self.w[:,-2:-1,:], self.w[:,2:-1,:], self.w[:,-2:-1,:]], axis = 1);
  def indices(self, indices_x, indices_y, indices_z, dx = 0, dy = 0, dz = 0):
    return tf.stack([indices_x + dx, indices_y + dy, indices_z + dz], axis = -1);
  def momento_x(self, u_old, v_old, w_old, velocity_iter):
    indices_x = tf.tile(tf.reshape(tf.range(2, self.nx), (-1, 1, 1)), (1, self.ny - 1, self.nz - 1)); # indices_x = 2, ... , nx - 1 has totally nx - 2 numbers
    indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 2, 1, self.nz - 1)); # indices_y = 1, ... , ny - 1 has totally ny - 1 numbers
    indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (self.nx - 2, self.ny - 1, 1)); # indices_z = 1, ... , nz - 1 has totally nz - 1 numbers
    indices = tf.stack([indices_x, indices_y, indices_z], axis = -1); # indices.shape = (nx - 2, ny - 1, nz - 1, 3)
    # areas
    area_east = (tf.gather(self.x, indices[...,0]) + tf.gather(self.x, indices[...,0] + 1)) / 2 * \
                (tf.gather(self.y, indices[...,1] + 1) - tf.gather(self.y, indices[...,1])) * \
                (tf.gather(self.z, indices[...,2]) + tf.gather(self.z, indices[...,2] + 1)) / 2;
    area_west = (tf.gather(self.x, indices[...,0]) + tf.gather(self.x, indices[...,0] - 1)) / 2 * \
                (tf.gather(self.y, indices[...,1] + 1) - tf.gather(self.y, indices[...,1])) * \
                (tf.gather(self.z, indices[...,2]) + tf.gather(self.z, indices[...,2] + 1)) / 2;
    area_north = (tf.gather(self.x, indices[...,0] + 1) - tf.gather(self.x, indices[...,0] + 1)) / 2 * \
                 (tf.gather(self.z, indices[...,2]) + tf.gather(self.z, indices[...,2] + 1)) / 2;
    area_south = (tf.gather(self.x, indices[...,0] + 1) - tf.gather(self.x, indices[...,0] + 1)) / 2 * \
                 (tf.gather(self.z, indices[...,2]) + tf.gather(self.z, indices[...,2] + 1)) / 2;
    area_top = (tf.gather(self.y, indices[...,1] + 1) - tf.gather(self.y, indices[...,1])) * \
               (((tf.gather(self.x, indices[...,0]) + tf.gather(self.x, indices[...,0] + 1)) / 2)**2 - ((tf.gather(self.x, indices[...,0]) + tf.gather(self.x, indices[...,0] - 1)) / 2)**2) / 2;
    area_bottom = (tf.gather(self.y, indices[...,1] + 1) - tf.gather(self.y, indices[...,1])) * \
                  (((tf.gather(self.x, indices[...,0]) + tf.gather(self.x, indices[...,0] + 1)) / 2)**2 - ((tf.gather(self.x, indices[...,0]) + tf.gather(self.x, indices[...,0] - 1)) / 2)**2) / 2;
    # flows
    flow_east = .5 * self.rho * area_east * (tf.gather_nd(u_old, self.indices(indices_x + 1, indices_y, indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(indices_x, indices_y, indices_z)));
    flow_west = .5 * self.rho * area_west * (tf.gather_nd(u_old, self.indices(indices_x - 1, indices_y, indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(indices_x, indices_y, indices_z)));
    flow_north = .5 * self.rho * area_north * (tf.gather_nd(v_old, self.indices(indices_x - 1, indices_y + 1, indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(indices_x, indices_y + 1, indices_z)));
    flow_south = .5 * self.rho * area_south * (tf.gather_nd(v_old, self.indices(indices_x - 1, indices_y, indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(indices_x, indices_y, indices_z)));
    flow_top = .5 * self.rho * area_top * (tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z + 1)) + \
                                           tf.gather_nd(w_old, self.indices(indices_x - 1, indices_y, indices_z + 1)));
    flow_bottom = .5 * self.rho * area_bottom * (tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z)) + \
                                                 tf.gather_nd(w_old, self.indices(indices_x - 1, indices_y, indices_z)));
    # system coefficients
    Ae = tf.math.maximum(-flow_east, 0);
    Aw = tf.math.maximum(flow_west, 0);
    An = tf.math.maximum(-flow_north, 0);
    As = tf.math.maximum(flow_south, 0);
    At = tf.math.maximum(-flow_top, 0);
    Ab = tf.math.maximum(flow_bottom, 0);
    Apu = Ae + Aw + An + As + At + Ab;
    Dcu = -(Ae * tf.gather_nd(self.u, self.indices(indices_x + 1, indices_y, indices_z)) + \
            Aw * tf.gather_nd(self.u, self.indices(indices_x - 1, indices_y, indices_z)) + \
            An * tf.gather_nd(self.u, self.indices(indices_x, indices_y + 1, indices_z)) + \
            As * tf.gather_nd(self.u, self.indices(indices_x, indices_y - 1, indices_z)) + \
            At * tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z + 1)) + \
            Ab * tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z - 1)) + \
            Apu * tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z)));
    Dcc = .5 * (flow_east * (tf.gather_nd(self.u, self.indices(indices_x + 1, indices_y, indices_z)) + tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z))) - \
                flow_west * (tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(self.u, self.indices(indices_x - 1, indices_y, indices_z))) + \
                flow_north * (tf.gather_nd(self.u, self.indices(indices_x, indices_y + 1, indices_z)) + tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z))) - \
                flow_south * (tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(self.u, self.indices(indices_x, indices_y - 1, indices_z))) + \
                flow_top * (tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z + 1)) + tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z))) - \
                flow_bottom * (tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z - 1))));
    Ae += self.mu * area_east / (tf.gather(self.x, indices_x + 1) - tf.gather(self.x, indices_x));
    Aw += self.mu * area_west / (tf.gather(self.x, indices_x) - tf.gather(self.x, indices_x - 1));
    An += self.mu * area_north / ((tf.gather(self.y, indices_y + 1) - tf.gather(self.y, indices_y)) * tf.gather(self.x, indices_x));
    As += self.mu * area_south / ((tf.gather(self.y, indices_y) - tf.gather(self.y, indices_y - 1)) * tf.gather(self.x, indices_x));
    At += self.mu * area_top / (tf.gather(self.z, indices_z + 1) - tf.gather(self.z, indices_z));
    Ab += self.mu * area_bottom / (tf.gather(self.z, indices_z) - tf.gather(self.z, indices_z - 1));
    # outline
    area_north = (tf.gather(self.x, indices_x + 1) - tf.gather(self.x, indices_x + 1)) / 2 * (tf.gather(self.z, indices_z) + tf.gather(self.z, indices_z + 1)) / 2;
    area_south = (tf.gather(self.x, indices_x + 1) - tf.gather(self.x, indices_x + 1)) / 2 * (tf.gather(self.z, indices_z) + tf.gather(self.z, indices_z + 1)) / 2;
    flow_north = .5 * self.rho * area_north * (tf.gather_nd(v_old, self.indices(indices_x - 1, self.ny * tf.ones_like(indices_y, dtype = tf.int32), indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(indices_x, self.ny * tf.ones_like(indices_y, dtype = tf.int32), indices_z)));
    flow_south = .5 * self.rho * area_south * (tf.gather_nd(v_old, self.indices(indices_x - 1, tf.ones_like(indices_y, dtype = tf.int32), indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(indices_x, tf.ones_like(indices_y, dtype = tf.int32), indices_z)));
    tail = tf.math.maximum(-flow_north, 0) + self.mu * area_north / ((tf.gather(self.y, indices_y + 1) - tf.gather(self.y, indices_y)) * tf.gather(self.x, indices_x) / 2);
    head = tf.math.maximum(flow_south, 0) + self.mu * area_south / ((tf.gather(self.y, tf.ones_like(indices_y)) - tf.gather(self.y, tf.zeros_like(indices_y))) * tf.gather(self.x, indices_x) / 2);
    An = tf.concat([An[:,:-1,:], tail[:,-1:,:]], axis = 1); # An.shape = (nx-2, ny-1, nz-1)
    As = tf.concat([head[:,:1,:], As[:,1:,:]], axis = 1); # As.shape = (nx-2, ny-1, nz-1)
    
    area_top = (tf.gather(self.y, indices_y + 1) - tf.gather(self.y, indices_y)) * \
               (((tf.gather(self.x, indices_x) + tf.gather(self.x, indices_x + 1))/2)**2 - \
                ((tf.gather(self.x, indices_x) + tf.gather(self.x, indices_x - 1))/2)**2) / 2;
    area_bottom = (tf.gather(self.y, indices_y + 1) - tf.gather(self.y, indices_y)) * \
                  (((tf.gather(self.x, indices_x) + tf.gather(self.x, indices_x + 1))/2)**2 - \
                   ((tf.gather(self.x, indices_x) + tf.gather(self.x, indices_x - 1))/2)**2) / 2;
    flow_top = .5 * self.rho * self.dx * self.dy * (tf.gather_nd(w_old, self.indices(indices_x, indices_y - 1, self.nz * tf.ones_like(indices_z, dtype = tf.int32))) + \
                                                    tf.gather_nd(w_old, self.indices(indices_x, indices_y, self.nz * tf.ones_like(indices_z))));
    flow_bottom = .5 * self.rho * self.dx * self.dy * (tf.gather_nd(w_old, self.indices(indices_x, indices_y - 1, tf.ones_like(indices_z, dtype = tf.int32))) + \
                                                       tf.gather_nd(w_old, self.indices(indices_x, indices_y, tf.ones_like(indices_z, dtype = tf.int32))));
    tail = tf.math.maximum(-flow_top, 0) + self.mu * area_top / ((tf.gather(self.z, indices_z + 1) - tf.gather(self.z, indices_z)) / 2);
    head = tf.math.maximum(flow_bottom, 0) + self.mu * area_bottom / ((tf.gather(self.z, tf.ones_like(indices_z)) - tf.gather(self.z, tf.zeros_like(indices_z))) / 2);
    At = tf.concat([At[:,:,:-1], tail[:,:,-1:]], axis = 2); # An.shape = (nx-2, ny-1, nz-1)
    Ab = tf.concat([head[:,:,:1], Ab[:,:,1:]], axis = 2); # Ab.shape = (nx-2, ny-1, nz-1)
    # Calculation
    Apu = (Ae + Aw + An + As + At + Ab) / self.omega_u;
    # update self.u with iteration
    for i in range(velocity_iter):
      dV = -(tf.gather(self.y, indices_y - 1) - tf.gather(self.y, indices_y + 1)) * \
            (tf.gather(self.z, indices_z - 1) - tf.gather(self.z, indices_z + 1)) * \
            (2 * tf.gather(self.x, indices_x) + tf.gather(self.x, indices_x - 1) + tf.gather(self.x, indices_x + 1)) * \
            (tf.gather(self.x, indices_x - 1) - tf.gather(self.x, indices_x + 1)) / 32;
      dX = (tf.gather(self.x, indices_x + 1) - tf.gather(self.x, indices_x - 1)) / 2;
      Valor = Ae * tf.gather_nd(self.u, self.indices(indices_x + 1, indices_y, indices_z)) + \
              Aw * tf.gather_nd(self.u, self.indices(indices_x - 1, indices_y, indices_z)) + \
              An * tf.gather_nd(self.u, self.indices(indices_x, indices_y + 1, indices_z)) + \
              As * tf.gather_nd(self.u, self.indices(indices_x, indices_y - 1, indices_z)) + \
              At * tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z + 1)) + \
              Ab * tf.gather_nd(self.u, self.indices(indices_x, indices_y, indices_z - 1)) - \
              self.beta * (Dcc - Dcu) + \
              dV / dX * (tf.gather_nd(self.P, self.indices(indices_x - 1, indices_y, indices_z)) - \
                         tf.gather_nd(self.P, self.indices(indices_x, indices_y, indices_z)));
      self.u = (1 - self.omega_u) * u_old + tf.pad(Valor / Apu, [[2,1],[1,1],[1,1]]);
    Apu = tf.pad(Apu, [[2,1],[1,1],[1,1]]);
    return Apu;
  def momento_y(self,):
    pass;
  def momento_z(self,):
    pass;
  def solve(self, iteration = 10, velocity_iter = 10, pressure_iter = 20):
    u_old, v_old, w_old = self.u, self.v, self.w;
    Apu = self.momento_x(u_old, v_old, w_old, velocity_iter);
    Apv = self.momento_y(u_old, v_old, w_old, velocity_iter);
    Apw = self.momento_z(u_old, v_old, w_old, velocity_iter);

if __name__ == "__main__":
  simple = SIMPLE();
  simple.solve();

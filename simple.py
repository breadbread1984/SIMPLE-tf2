#!/usr/bin/python3

from math import pi;
import numpy as np;
import tensorflow as tf;

class SIMPLE(object):
  def __init__(self, nx=20,ny=30,nz=20, dtype = tf.float64):
    self.dtype = dtype;
    self.nx = nx;
    self.ny = ny;
    self.nz = nz;
    # domain discretization
    self.x, self.y, self.z, self.dx, self.dy, self.dz = self.domain_discretization();
    # fluid properties
    self.rho, self.mu = self.fluid_properties();
    # underelaxation properties
    self.omega_u, self.omega_v, self.omega_w, self.omega_p, self.omega_pp, self.beta = self.underelaxation_properties();
    # create computing graphs
    self.initialization_graph = tf.Graph();
    with self.initialization_graph.as_default():
      self.initialization_outputs = self.initialization();
    self.set_conditions_graph = tf.Graph();
    with self.set_conditions_graph.as_default():
      self.set_conditions_inputs, self.set_conditions_outputs = self.set_conditions();
    self.momento_x_graph = tf.Graph();
    with self.momento_x_graph.as_default():
      self.momento_x_inputs, self.momento_x_outputs = self.momento_x(10);
    self.momento_y_graph = tf.Graph();
    with self.momento_y_graph.as_default():
      self.momento_y_inputs, self.momento_y_outputs = self.momento_y(10);
    self.momento_z_graph = tf.Graph();
    with self.momento_z_graph.as_default():
      self.momento_z_inputs, self.momento_z_outputs = self.momento_z(10);
    self.pressure_graph = tf.Graph();
    with self.pressure_graph.as_default():
      self.pressure_inputs, self.pressure_outputs = self.pressure(20);
    self.ensure_quality_graph = tf.Graph();
    with self.ensure_quality_graph.as_default():
      self.ensure_quality_inputs, self.ensure_quality_outputs = self.ensure_quality();
    self.error_source_graph = tf.Graph();
    with self.error_source_graph.as_default():
      self.error_source_inputs, self.error_source_outputs = self.error_source();
    # velocities, pressure initialization
    with tf.Session(graph = self.initialization_graph) as sess:
      self.u, self.v, self.w, self.P = sess.run([output.outputs[0] for output in self.initialization_outputs]);
    # set conditions
    with tf.Session(graph = self.set_conditions_graph) as sess:
      self.u, self.v, self.w, self.P = sess.run([output.outputs[0] for output in self.set_conditions_outputs],
                                                feed_dict = {input.outputs[0]:value for input, value in zip(self.set_conditions_inputs, [self.u, self.v, self.w, self.P, self.x, self.y, self.z])});

  def domain_discretization(self,):
    dx = 1/self.nx;
    x = np.array([i * dx for i in range(self.nx + 1)], dtype = np.float32 if self.dtype == tf.float32 else np.float64); # x in range [0,1] quantized into nx+1 values
    dy = 2*pi/(self.ny + 1);
    y = np.array([i * dy for i in range(self.ny + 1)], dtype = np.float32 if self.dtype == tf.float32 else np.float64); # y in range [0, 2 * pi] quantized into ny+1 values
    dz = 1/self.nz;
    z = np.array([i * dz for i in range(self.nz + 1)], dtype = np.float32 if self.dtype == tf.float32 else np.float64); # z in range [0,1] quantized into nz+1 values
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
    u = tf.zeros((self.nx+1, self.ny+1, self.nz+1), dtype = self.dtype); # u.shape = (21, 31, 21)
    v = tf.scatter_nd(indices = indices, updates = tf.ones((self.nx-1,self.ny-1,self.nz-1), dtype = self.dtype), shape = (self.nx+1,self.ny+1,self.nz+1)); # v.shape = (21, 31, 21)
    w = tf.zeros((self.nx+1, self.ny+1, self.nz+1), dtype = self.dtype); # w.shape = (21, 31, 21)
    P = tf.zeros((self.nx+1, self.ny+1, self.nz+1), dtype = self.dtype); # P.shape = (21, 31, 21)
    return u,v,w,P;
  def set_conditions(self,):
    u = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    v = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    w = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    P = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    x = tf.placeholder(shape = (self.nx + 1), dtype = self.dtype);
    y = tf.placeholder(shape = (self.ny + 1), dtype = self.dtype);
    z = tf.placeholder(shape = (self.nz + 1), dtype = self.dtype);
    # NOTE: derivable member function
    indices_x = tf.tile(tf.reshape(tf.range(1, self.nx + 1), (-1, 1, 1)), (1, self.ny + 1, self.nz)); # indices_x.shape = (nx, ny+1, nz)
    indices_y = tf.tile(tf.reshape(tf.range(0, self.ny + 1), (1, -1, 1)), (self.nx, 1, self.nz)); # indices_y.shape = (nx, ny+1, nz)
    indices_z = tf.tile(tf.reshape(tf.range(1, self.nz + 1), (1, 1, -1)), (self.nx, self.ny + 1, 1)); # indices_z.shape = (nx, ny+1. nz)
    loop_mask = tf.scatter_nd(tf.reshape(self.indices(indices_x, indices_y, indices_z), (-1, 3)), tf.ones((self.nx * (self.ny + 1) * self.nz,), dtype = tf.bool), (self.nx + 1, self.ny + 1, self.nz + 1)); # loop_mask.shape = (nx+1, ny+1, nz+1)
    x_less_mask = tf.tile(tf.reshape(tf.math.less_equal(x, 0.4), (-1, 1, 1)), (1, self.ny + 1, self.nz + 1));
    z_greater_mask = tf.tile(tf.reshape(tf.math.greater_equal(z, 0.2), (1, 1, -1)), (self.nx + 1, self.ny + 1, 1));
    mask = tf.math.logical_and(tf.math.logical_and(x_less_mask, z_greater_mask), loop_mask);
    tmp_u = tf.where(mask, tf.zeros_like(u), u);
    tmp_v = tf.where(mask, 2.5 * tf.tile(tf.reshape(x, (-1, 1, 1)), (1, self.ny + 1, self.nz + 1)), v);
    tmp_w = tf.where(mask, tf.zeros_like(w), w);

    new_u = tf.concat([tmp_u[:,-2:-1,:], tmp_u[:,-2:-1,:], tmp_u[:,2:-1,:], tmp_u[:,-2:-1,:]], axis = 1);
    new_v = tf.concat([tmp_v[:,-2:-1,:], tmp_v[:,-2:-1,:], tmp_v[:,2:-1,:], tmp_v[:,-2:-1,:]], axis = 1);
    new_w = tf.concat([tmp_w[:,-2:-1,:], tmp_w[:,-2:-1,:], tmp_w[:,2:-1,:], tmp_w[:,-2:-1,:]], axis = 1);
    new_P = tf.concat([P[:,-1:,:], P[:,1:,:]], axis = 1);
    return (u, v, w, P, x, y, z), (new_u, new_v, new_w, new_P);
  def indices(self, indices_x, indices_y, indices_z, dx = 0, dy = 0, dz = 0):
    return tf.stack([indices_x + dx, indices_y + dy, indices_z + dz], axis = -1);
  def momento_x(self, velocity_iter):
    u = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    P = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    u_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    v_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    w_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    x = tf.placeholder(shape = (self.nx + 1), dtype = self.dtype);
    y = tf.placeholder(shape = (self.ny + 1), dtype = self.dtype);
    z = tf.placeholder(shape = (self.nz + 1), dtype = self.dtype);
    
    indices_x = tf.tile(tf.reshape(tf.range(2, self.nx), (-1, 1, 1)), (1, self.ny - 1, self.nz - 1)); # indices_x = 2, ... , nx - 1 has totally nx - 2 numbers
    indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 2, 1, self.nz - 1)); # indices_y = 1, ... , ny - 1 has totally ny - 1 numbers
    indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (self.nx - 2, self.ny - 1, 1)); # indices_z = 1, ... , nz - 1 has totally nz - 1 numbers
    # areas
    area_east = (tf.gather(x, indices_x) + tf.gather(x, indices_x + 1)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
                (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_west = (tf.gather(x, indices_x) + tf.gather(x, indices_x - 1)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
                (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_north = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x + 1)) / 2 * \
                 (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_south = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x + 1)) / 2 * \
                 (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_top = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
               (((tf.gather(x, indices_x) + tf.gather(x, indices_x + 1)) / 2)**2 - \
                ((tf.gather(x, indices_x) + tf.gather(x, indices_x - 1)) / 2)**2) / 2;
    area_bottom = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
                  (((tf.gather(x, indices_x) + tf.gather(x, indices_x + 1)) / 2)**2 - \
                   ((tf.gather(x, indices_x) + tf.gather(x, indices_x - 1)) / 2)**2) / 2;
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
    Dcu = -(Ae * tf.gather_nd(u, self.indices(indices_x + 1, indices_y, indices_z)) + \
            Aw * tf.gather_nd(u, self.indices(indices_x - 1, indices_y, indices_z)) + \
            An * tf.gather_nd(u, self.indices(indices_x, indices_y + 1, indices_z)) + \
            As * tf.gather_nd(u, self.indices(indices_x, indices_y - 1, indices_z)) + \
            At * tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z + 1)) + \
            Ab * tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z - 1))) + \
          Apu * tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z));
    Dcc = .5 * (flow_east * (tf.gather_nd(u, self.indices(indices_x + 1, indices_y, indices_z)) + tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z))) - \
                flow_west * (tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(u, self.indices(indices_x - 1, indices_y, indices_z))) + \
                flow_north * (tf.gather_nd(u, self.indices(indices_x, indices_y + 1, indices_z)) + tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z))) - \
                flow_south * (tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(u, self.indices(indices_x, indices_y - 1, indices_z))) + \
                flow_top * (tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z + 1)) + tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z))) - \
                flow_bottom * (tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z - 1))));
    Ae += self.mu * area_east / (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x));
    Aw += self.mu * area_west / (tf.gather(x, indices_x) - tf.gather(x, indices_x - 1));
    An += self.mu * area_north / ((tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * tf.gather(x, indices_x));
    As += self.mu * area_south / ((tf.gather(y, indices_y) - tf.gather(y, indices_y - 1)) * tf.gather(x, indices_x));
    At += self.mu * area_top / (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z));
    Ab += self.mu * area_bottom / (tf.gather(z, indices_z) - tf.gather(z, indices_z - 1));
    # outline
    ns_indices_x = tf.tile(tf.reshape(tf.range(2, self.nx), (-1, 1, 1)), (1, 1, self.nz - 1)); # ns_indices_x.shape = (nx-2, 1, nz-1)
    ns_head_indices_y = tf.tile(tf.reshape([1], (1, -1, 1)), (self.nx - 2, 1, self.nz - 1)); # ns_head_indices_y.shape = (nx-2, 1, nz-1)
    ns_tail_indices_y = tf.tile(tf.reshape([self.ny], (1, -1, 1)), (self.nx - 2, 1, self.nz - 1)); # ns_tail_indices_y.shape = (nx-2, 1, nz-1)
    ns_indices_z = tf.tile(tf.reshape(tf.range(1,self.nz), (1, 1, -1)), (self.nx - 2, 1, 1)); # ns_indices_z.shape = (nx-2, 1, nz-1)
    area_north = (tf.gather(x, ns_indices_x + 1) - tf.gather(x, ns_indices_x + 1)) / 2 * \
                 (tf.gather(z, ns_indices_z) + tf.gather(z, ns_indices_z + 1)) / 2;
    area_south = (tf.gather(x, ns_indices_x + 1) - tf.gather(x, ns_indices_x + 1)) / 2 * \
                 (tf.gather(z, ns_indices_z) + tf.gather(z, ns_indices_z + 1)) / 2;
    flow_north = .5 * self.rho * area_north * (tf.gather_nd(v_old, self.indices(ns_indices_x - 1, ns_tail_indices_y, ns_indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(ns_indices_x, ns_tail_indices_y, ns_indices_z)));
    flow_south = .5 * self.rho * area_south * (tf.gather_nd(v_old, self.indices(ns_indices_x - 1, ns_head_indices_y, ns_indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(ns_indices_x, ns_head_indices_y, ns_indices_z)));
    An_tail = tf.math.maximum(-flow_north, 0) + self.mu * area_north / ((tf.gather(y, ns_tail_indices_y) - tf.gather(y, ns_tail_indices_y - 1)) * tf.gather(x, ns_indices_x) / 2); # An_tail.shape = (nx-2, 1, nz-1)
    As_head = tf.math.maximum(flow_south, 0) + self.mu * area_south / ((tf.gather(y, ns_head_indices_y) - tf.gather(y, ns_head_indices_y - 1)) * tf.gather(x, ns_indices_x) / 2); # An_head.shape = (nx-2, 1, nz-1)
    An = tf.concat([An[:,:-1,:], An_tail], axis = 1); # An.shape = (nx-2, ny-1, nz-1)
    As = tf.concat([As_head, As[:,1:,:]], axis = 1); # As.shape = (nx-2, ny-1, nz-1)

    tb_indices_x = tf.tile(tf.reshape(tf.range(2, self.nx), (-1, 1, 1)), (1, self.ny - 1, 1)); # tb_indices_x.shape = (nx-2, ny-1, 1)
    tb_indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 2, 1, 1)); # tb_indices_y.shape = (nx-2, ny-1, 1)
    tb_head_indices_z = tf.tile(tf.reshape([1], (1, 1, -1)), (self.nx - 2, self.ny - 1, 1)); # tb_head_indices_z.shape = (nx-2, ny-1, 1)
    tb_tail_indices_z = tf.tile(tf.reshape([self.nz], (1, 1, -1)), (self.nx - 2, self.ny - 1, 1)); # tb_tail_indices_z.shape = (nx-2, ny-1, 1)
    area_top = (tf.gather(y, tb_indices_y + 1) - tf.gather(y, tb_indices_y)) * \
               (((tf.gather(x, tb_indices_x) + tf.gather(x, tb_indices_x + 1))/2)**2 - \
                ((tf.gather(x, tb_indices_x) + tf.gather(x, tb_indices_x - 1))/2)**2) / 2;
    area_bottom = (tf.gather(y, tb_indices_y + 1) - tf.gather(y, tb_indices_y)) * \
                  (((tf.gather(x, tb_indices_x) + tf.gather(x, tb_indices_x + 1))/2)**2 - \
                   ((tf.gather(x, tb_indices_x) + tf.gather(x, tb_indices_x - 1))/2)**2) / 2;
    flow_top = .5 * self.rho * self.dx * self.dy * (tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y - 1, tb_tail_indices_z)) + \
                                                    tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y, tb_tail_indices_z)));
    flow_bottom = .5 * self.rho * self.dx * self.dy * (tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y - 1, tb_head_indices_z)) + \
                                                       tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y, tb_head_indices_z)));
    At_tail = tf.math.maximum(-flow_top, 0) + self.mu * area_top / ((tf.gather(z, tb_tail_indices_z) - tf.gather(z, tb_tail_indices_z - 1)) / 2);
    Ab_head = tf.math.maximum(flow_bottom, 0) + self.mu * area_bottom / ((tf.gather(z, tb_head_indices_z) - tf.gather(z, tb_head_indices_z - 1)) / 2);
    At = tf.concat([At[:,:,:-1], At_tail], axis = 2); # An.shape = (nx-2, ny-1, nz-1)
    Ab = tf.concat([Ab_head, Ab[:,:,1:]], axis = 2); # Ab.shape = (nx-2, ny-1, nz-1)
    # Calculation
    Apu = (Ae + Aw + An + As + At + Ab) / self.omega_u; # Apu.shape = (nx - 2, ny - 1, nz - 1)
    Apu = tf.pad(Apu, [[2,1],[1,1],[1,1]]); # Apu.shap = (nx + 1, ny + 1, nz + 1)
    # update self.u with iteration
    for i in range(velocity_iter):
      dV = -(tf.gather(y, indices_y - 1) - tf.gather(y, indices_y + 1)) * \
            (tf.gather(z, indices_z - 1) - tf.gather(z, indices_z + 1)) * \
            (2 * tf.gather(x, indices_x) + tf.gather(x, indices_x - 1) + tf.gather(x, indices_x + 1)) * \
            (tf.gather(x, indices_x - 1) - tf.gather(x, indices_x + 1)) / 32;
      dX = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x - 1)) / 2;
      Valor = Ae * tf.gather_nd(u, self.indices(indices_x + 1, indices_y, indices_z)) + \
              Aw * tf.gather_nd(u, self.indices(indices_x - 1, indices_y, indices_z)) + \
              An * tf.gather_nd(u, self.indices(indices_x, indices_y + 1, indices_z)) + \
              As * tf.gather_nd(u, self.indices(indices_x, indices_y - 1, indices_z)) + \
              At * tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z + 1)) + \
              Ab * tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z - 1)) - \
              self.beta * (Dcc - Dcu) + \
              dV / dX * (tf.gather_nd(P, self.indices(indices_x - 1, indices_y, indices_z)) - \
                         tf.gather_nd(P, self.indices(indices_x, indices_y, indices_z)));
      u_update = (1 - self.omega_u) * u_old + tf.pad(Valor, [[2,1],[1,1],[1,1]]) / Apu;
      new_u = tf.where(
                 tf.cast(tf.scatter_nd(
                   tf.reshape(self.indices(indices_x, indices_y, indices_z), (-1, 3)),
                   tf.ones(((self.nx - 2) * (self.ny - 1) * (self.nz - 1),)),
                   (self.nx + 1, self.ny + 1, self.nz + 1)), dtype = tf.bool),
                 tf.scatter_nd(
                   tf.reshape(self.indices(indices_x, indices_y, indices_z), (-1, 3)),
                   tf.reshape(u_update[2:-1,1:-1,1:-1], (-1,)),
                   (self.nx + 1, self.ny + 1, self.nz + 1)),
                 u
               );
      #assert tf.math.reduce_any(tf.math.is_inf(self.u)) != True;
      #assert tf.math.reduce_any(tf.math.is_nan(self.u)) != True;
    return (u, P, u_old, v_old, w_old, x, y, z), (new_u, Apu);
  def momento_y(self, velocity_iter):
    v = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    P = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    u_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    v_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    w_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    x = tf.placeholder(shape = (self.nx + 1), dtype = self.dtype);
    y = tf.placeholder(shape = (self.ny + 1), dtype = self.dtype);
    z = tf.placeholder(shape = (self.nz + 1), dtype = self.dtype);
    
    indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, self.ny - 2, self.nz - 1)); # indices_x = 1, ... , nx - 1 has totally nx - 1 numbers
    indices_y = tf.tile(tf.reshape(tf.range(2, self.ny), (1, -1, 1)), (self.nx - 1, 1, self.nz - 1)); # indices_y = 2, ... , ny - 1 has totally ny - 2 numbers
    indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (self.nx - 1, self.ny - 2, 1)); # indices_z = 1, ... , nz - 1 has totally nz - 1 numbers
    # areas
    area_east = tf.gather(x, indices_x + 1) * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_west = tf.gather(x, indices_x) * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_north = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) * \
                 (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_south = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) * \
                 (tf.gather(z, indices_z) + tf.gather(z, indices_z + 1)) / 2;
    area_top = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
               (tf.gather(x, indices_x + 1)**2 - tf.gather(x, indices_x - 1)**2) / 2;
    area_bottom = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                  (tf.gather(x, indices_x + 1)**2 - tf.gather(x, indices_x - 1)**2) / 2;
    # flows
    flow_east = .5 * self.rho * area_east * (tf.gather_nd(u_old, self.indices(indices_x + 1, indices_y, indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(indices_x + 1, indices_y - 1, indices_z)));
    flow_west = .5 * self.rho * area_west * (tf.gather_nd(u_old, self.indices(indices_x, indices_y - 1, indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(indices_x, indices_y, indices_z)));
    flow_north = .5 * self.rho * area_north * (tf.gather_nd(v_old, self.indices(indices_x, indices_y + 1, indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(indices_x, indices_y, indices_z)));
    flow_south = .5 * self.rho * area_south * (tf.gather_nd(v_old, self.indices(indices_x, indices_y, indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(indices_x, indices_y - 1, indices_z)));
    flow_top = .5 * self.rho * area_top * (tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z + 1)) + \
                                           tf.gather_nd(w_old, self.indices(indices_x, indices_y - 1, indices_z + 1)));
    flow_bottom = .5 * self.rho * area_bottom * (tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z)) + \
                                                 tf.gather_nd(w_old, self.indices(indices_x, indices_y - 1, indices_z)));
    # system coefficients
    Ae = tf.math.maximum(-flow_east, 0);
    Aw = tf.math.maximum(flow_west, 0);
    An = tf.math.maximum(-flow_north, 0);
    As = tf.math.maximum(flow_south, 0);
    At = tf.math.maximum(-flow_top, 0);
    Ab = tf.math.maximum(flow_bottom, 0);
    Apv = Ae + Aw + An + As + At + Ab;
    Dcu = -(Ae * tf.gather_nd(v, self.indices(indices_x + 1, indices_y, indices_z)) + \
            Aw * tf.gather_nd(v, self.indices(indices_x - 1, indices_y, indices_z)) + \
            An * tf.gather_nd(v, self.indices(indices_x, indices_y + 1, indices_z)) + \
            As * tf.gather_nd(v, self.indices(indices_x, indices_y - 1, indices_z)) + \
            At * tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z + 1)) + \
            Ab * tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z - 1))) + \
          Apv * tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z));
    Dcc = .5 * (flow_east * (tf.gather_nd(v, self.indices(indices_x + 1, indices_y, indices_z)) + tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z))) - \
                flow_west * (tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(v, self.indices(indices_x - 1, indices_y, indices_z))) + \
                flow_north * (tf.gather_nd(v, self.indices(indices_x, indices_y + 1, indices_z)) + tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z))) - \
                flow_south * (tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(v, self.indices(indices_x, indices_y - 1, indices_z))) + \
                flow_top * (tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z + 1)) + tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z))) - \
                flow_bottom * (tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z - 1))));
    Ae += self.mu * area_east / (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x));
    Aw += self.mu * area_west / (tf.gather(x, indices_x) - tf.gather(x, indices_x - 1));
    An += self.mu * area_north / ((tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * tf.gather(x, indices_x));
    As += self.mu * area_south / ((tf.gather(y, indices_y) - tf.gather(y, indices_y - 1)) * tf.gather(x, indices_x));
    At += self.mu * area_top / (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z));
    Ab += self.mu * area_bottom / (tf.gather(z, indices_z) - tf.gather(z, indices_z - 1));
    # outline
    ew_head_indices_x = tf.tile(tf.reshape([1], (-1, 1, 1)), (1, self.ny - 2, self.nz - 1)); # ew_head_indices_x.shape = (1, ny-2, nz-1)
    ew_tail_indices_x = tf.tile(tf.reshape([self.nx], (-1, 1, 1)), (1, self.ny - 2, self.nz - 1)); # w_tail_indices_x.shape = (1. ny-2. nz-1)
    ew_indices_y = tf.tile(tf.reshape(tf.range(2, self.ny), (1, -1, 1)), (1, 1, self.nz - 1)); # ew_indices_y.shape = (1, ny-2, nz-1)
    ew_indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (1, self.ny - 2, 1)); # ew_indices_z.shape = (1, ny-2, nz-1)
    area_east = tf.gather(x, ew_tail_indices_x) * (tf.gather(y, ew_indices_y + 1) - tf.gather(y, ew_indices_y - 1)) / 2 * (tf.gather(z, ew_indices_z) + tf.gather(z, ew_indices_z + 1)) / 2;
    area_west = tf.gather(x, ew_head_indices_x) * (tf.gather(y, ew_indices_y + 1) - tf.gather(y, ew_indices_y - 1)) / 2 * (tf.gather(z, ew_indices_z) + tf.gather(z, ew_indices_z + 1)) / 2;
    flow_east = .5 * self.rho * area_east * (tf.gather_nd(u_old, self.indices(ew_tail_indices_x, ew_indices_y, ew_indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(ew_tail_indices_x, ew_indices_y - 1, ew_indices_z)));
    flow_west = .5 * self.rho * area_west * (tf.gather_nd(u_old, self.indices(ew_head_indices_x, ew_indices_y - 1, ew_indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(ew_head_indices_x, ew_indices_y, ew_indices_z)));
    Ae_tail = tf.math.maximum(-flow_east, 0) + self.mu * area_east / ((tf.gather(x, ew_tail_indices_x) - tf.gather(x, ew_tail_indices_x - 1)) / 2);
    Aw_head = tf.math.maximum(flow_west, 0) + self.mu * area_west / ((tf.gather(x, ew_head_indices_x) - tf.gather(x, ew_head_indices_x - 1)) / 2);
    Ae = tf.concat([Ae[:-1,:,:], Ae_tail], axis = 0); # Ae_tail.shape = (nx-1, ny-2, nz-1)
    Aw = tf.concat([Aw_head, Aw[1:,:,:]], axis = 0); # Aw_head.shape = (nx-1, ny-2, nz-1)

    tb_indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, self.ny - 2, 1)); # tb_indices_x.shape = (nx-1, ny-2, 1)
    tb_indices_y = tf.tile(tf.reshape(tf.range(2, self.ny), (1, -1, 1)), (self.nx - 1, 1, 1)); # tb_indices_y.shape = (nx-1, ny-2, 1)
    tb_head_indices_z = tf.tile(tf.reshape([1], (1, 1, -1)), (self.nx - 1, self.ny - 2, 1)); # tb_head_indices_z.shape = (nx-1, ny-2, 1)
    tb_tail_indices_z = tf.tile(tf.reshape([self.nz], (1, 1, -1)), (self.nx - 1, self.ny - 2, 1)); # tb_tail_indices_z.shape = (nx-1, ny-2, 1)
    area_top = (tf.gather(y, tb_indices_y + 1) - tf.gather(y, tb_indices_y - 1)) / 2 * \
               (tf.gather(x, tb_indices_x + 1)**2 - tf.gather(x, tb_indices_x - 1)**2) / 2;
    area_bottom = (tf.gather(y, tb_indices_y + 1) - tf.gather(y, tb_indices_y - 1)) / 2 * \
                  (tf.gather(x, tb_indices_x + 1)**2 - tf.gather(x, tb_indices_x - 1)**2) / 2;
    flow_top = .5 * self.rho * area_top * (tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y - 1, tb_tail_indices_z)) + \
                                           tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y, tb_tail_indices_z)));
    flow_bottom = .5 * self.rho * area_bottom * (tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y - 1, tb_head_indices_z)) + \
                                                 tf.gather_nd(w_old, self.indices(tb_indices_x, tb_indices_y, tb_head_indices_z)));
    At_tail = tf.math.maximum(-flow_top, 0) + self.mu * area_top / ((tf.gather(z, tb_tail_indices_z) - tf.gather(z, tb_tail_indices_z - 1)) / 2);
    Ab_head = tf.math.maximum(flow_bottom, 0) + self.mu * area_bottom / ((tf.gather(z, tb_head_indices_z) - tf.gather(z, tb_head_indices_z - 1)) / 2);
    At = tf.concat([At[:,:,:-1], At_tail], axis = 2); # At_tail.shape = (nx-1, ny-2, nz-1)
    Ab = tf.concat([Ab_head, Ab[:,:,1:]], axis = 2); # Ab_head.shape = (nx-1, ny-2, 1)
    # Calculation
    Apv = (Ae + Aw + An + As + At + Ab) / self.omega_v; # Apv.shape = (nx-1, ny-2, nz-1)
    Apv = tf.pad(Apv, [[1,1],[2,1],[1,1]]); # Apv.shape = (nx + 1, ny + 1, nz + 1)
    # update self.v with iteration
    for i in range(velocity_iter):
      dV = -(tf.gather(y, indices_y - 1) - tf.gather(y, indices_y + 1)) * \
            (tf.gather(z, indices_z - 1) - tf.gather(z, indices_z + 1)) * \
            (2 * tf.gather(x, indices_x) + tf.gather(x, indices_x - 1) + tf.gather(x, indices_x + 1)) * \
            (tf.gather(x, indices_x - 1) - tf.gather(x, indices_x + 1)) / 32;
      dY = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2;
      Valor = Ae * tf.gather_nd(v, self.indices(indices_x + 1, indices_y, indices_z)) + \
              Aw * tf.gather_nd(v, self.indices(indices_x - 1, indices_y, indices_z)) + \
              An * tf.gather_nd(v, self.indices(indices_x, indices_y + 1, indices_z)) + \
              As * tf.gather_nd(v, self.indices(indices_x, indices_y - 1, indices_z)) + \
              At * tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z + 1)) + \
              Ab * tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z - 1)) - \
              self.beta * (Dcc - Dcu) + \
              dV / dY * (tf.gather_nd(P, self.indices(indices_x, indices_y - 1, indices_z)) - \
                         tf.gather_nd(P, self.indices(indices_x, indices_y, indices_z)));
      v_update = (1 - self.omega_v) * v_old + tf.pad(Valor, [[1,1],[2,1],[1,1]]) / Apv;
      new_v = tf.where(
                 tf.cast(tf.scatter_nd(
                   tf.reshape(self.indices(indices_x, indices_y, indices_z), (-1, 3)),
                   tf.ones(((self.nx - 1) * (self.ny - 2) * (self.nz - 1),)),
                   (self.nx + 1, self.ny + 1, self.nz + 1)), dtype = tf.bool),
                 tf.scatter_nd(
                   tf.reshape(self.indices(indices_x, indices_y, indices_z), (-1, 3)),
                   tf.reshape(v_update[1:-1,2:-1,1:-1], (-1,)),
                   (self.nx + 1, self.ny + 1, self.nz + 1)),
                 v
               );
      #assert tf.math.reduce_any(tf.math.is_inf(self.v)) != True;
      #assert tf.math.reduce_any(tf.math.is_nan(self.v)) != True;
    return (v, P, u_old, v_old, w_old, x, y, z), (new_v, Apv);
  def momento_z(self, velocity_iter):
    w = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    P = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    u_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    v_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    w_old = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    x = tf.placeholder(shape = (self.nx + 1), dtype = self.dtype);
    y = tf.placeholder(shape = (self.ny + 1), dtype = self.dtype);
    z = tf.placeholder(shape = (self.nz + 1), dtype = self.dtype);
    
    indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, self.ny - 1, self.nz - 2)); # indices_x = 1, ... , nx - 1 has totally nx - 1 numbers
    indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 1, 1, self.nz - 2)); # indices_y = 1, ... , ny - 1 has totally ny - 1 numbers
    indices_z = tf.tile(tf.reshape(tf.range(2, self.nz), (1, 1, -1)), (self.nx - 1, self.ny - 1, 1)); # indices_z = 2, ... , nz - 1 has totally nz - 2 numbers
    # areas
    area_east = tf.gather(x, indices_x + 1) * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_west = tf.gather(x, indices_x) * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_north = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_south = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_top = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
               (tf.gather(x, indices_x + 1)**2 - tf.gather(x, indices_x)**2) / 2;
    area_bottom = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * \
                  (tf.gather(x, indices_x + 1)**2 - tf.gather(x, indices_x)**2) / 2;
    # flows
    flow_east = .5 * self.rho * area_east * (tf.gather_nd(u_old, self.indices(indices_x + 1, indices_y, indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(indices_x + 1, indices_y, indices_z - 1)));
    flow_west = .5 * self.rho * area_west * (tf.gather_nd(u_old, self.indices(indices_x, indices_y, indices_z - 1)) + \
                                             tf.gather_nd(u_old, self.indices(indices_x, indices_y, indices_z)));
    flow_north = .5 * self.rho * area_north * (tf.gather_nd(v_old, self.indices(indices_x, indices_y + 1, indices_z)) + 
                                               tf.gather_nd(v_old, self.indices(indices_x, indices_y + 1, indices_z - 1)));
    flow_south = .5 * self.rho * area_south * (tf.gather_nd(v_old, self.indices(indices_x, indices_y, indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(indices_x, indices_y, indices_z - 1)));
    flow_top = .5 * self.rho * area_top * (tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z + 1)) + \
                                           tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z)));
    flow_bottom = .5 * self.rho * area_bottom * (tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z)) + \
                                                 tf.gather_nd(w_old, self.indices(indices_x, indices_y, indices_z - 1)));
    # system coefficients
    Ae = tf.math.maximum(-flow_east, 0);
    Aw = tf.math.maximum(flow_west, 0);
    An = tf.math.maximum(-flow_north, 0);
    As = tf.math.maximum(flow_south, 0);
    At = tf.math.maximum(-flow_top, 0);
    Ab = tf.math.maximum(flow_bottom, 0);
    Apw = Ae + Aw + An + As + At + Ab;
    Dcu = -(Ae * tf.gather_nd(w, self.indices(indices_x + 1, indices_y, indices_z)) + \
            Aw * tf.gather_nd(w, self.indices(indices_x - 1, indices_y, indices_z)) + \
            An * tf.gather_nd(w, self.indices(indices_x, indices_y + 1, indices_z)) + \
            As * tf.gather_nd(w, self.indices(indices_x, indices_y - 1, indices_z)) + \
            At * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z + 1)) + \
            Ab * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z - 1))) + \
          Apw * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z));
    Dcc = .5 * (flow_east * (tf.gather_nd(w, self.indices(indices_x + 1, indices_y, indices_z)) + tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z))) - \
                flow_west * (tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(w, self.indices(indices_x - 1, indices_y, indices_z))) + \
                flow_north * (tf.gather_nd(w, self.indices(indices_x, indices_y + 1, indices_z)) + tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z))) - \
                flow_south * (tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(w, self.indices(indices_x, indices_y - 1, indices_z))) + \
                flow_top * (tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z + 1)) + tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z))) - \
                flow_bottom * (tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z)) + tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z - 1))));
    Ae += self.mu * area_east / (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x));
    Aw += self.mu * area_west / (tf.gather(x, indices_x) - tf.gather(x, indices_x - 1));
    An += self.mu * area_north / ((tf.gather(y, indices_y + 1) - tf.gather(y, indices_y)) * tf.gather(x, indices_x));
    As += self.mu * area_south / ((tf.gather(y, indices_y) - tf.gather(y, indices_y - 1)) * tf.gather(x, indices_x));
    At += self.mu * area_top / (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z));
    Ab += self.mu * area_bottom / (tf.gather(z, indices_z) - tf.gather(z, indices_z - 1));
    # outline
    ns_indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, 1, self.nz - 2)); # ns_indices_x.shape = (nx-1, 1, nz-2)
    ns_tail_indices_y = tf.tile(tf.reshape([self.ny], (1, -1, 1)), (self.nx - 1, 1, self.nz - 2)); # ns_tail_indices_y.shape = (nx-1, 1, nz-2)
    ns_head_indices_y = tf.tile(tf.reshape([1], (1, -1, 1)), (self.nx - 1, 1, self.nz - 2)); # ns_head_indices_y.shape = (nx-1, 1, nz-2)
    ns_indices_z = tf.tile(tf.reshape(tf.range(2, self.nz), (1, 1, -1)), (self.nx - 1, 1, 1)); # ns_indices_z.shape = (nx-1, 1, nz-2)
    area_north = (tf.gather(x, ns_indices_x + 1) - tf.gather(x, ns_indices_x)) * (tf.gather(z, ns_indices_z + 1) - tf.gather(z, ns_indices_z - 1)) / 2;
    area_south = (tf.gather(x, ns_indices_x + 1) - tf.gather(x, ns_indices_x)) * (tf.gather(z, ns_indices_z + 1) - tf.gather(z, ns_indices_z - 1)) / 2;
    flow_north = .5 * self.rho * area_north * (tf.gather_nd(v_old, self.indices(ns_indices_x, ns_tail_indices_y, ns_indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(ns_indices_x, ns_tail_indices_y, ns_indices_z - 1)));
    flow_south = .5 * self.rho * area_south * (tf.gather_nd(v_old, self.indices(ns_indices_x, ns_head_indices_y, ns_indices_z)) + \
                                               tf.gather_nd(v_old, self.indices(ns_indices_x, ns_head_indices_y, ns_indices_z - 1)));
    An_tail = tf.math.maximum(-flow_north, 0) + self.mu * area_north / ((tf.gather(y, ns_tail_indices_y) - tf.gather(y, ns_tail_indices_y - 1)) * tf.gather(x, ns_indices_x) / 2);
    As_head = tf.math.maximum(flow_south, 0) + self.mu * area_south / ((tf.gather(y, ns_head_indices_y) - tf.gather(y, ns_head_indices_y - 1)) * tf.gather(x, ns_indices_x) / 2);
    An = tf.concat([An[:,:-1,:], An_tail], axis = 1); # An_tail.shape = (nx-1,ny-1,nz-2)
    As = tf.concat([As_head, As[:,1:,:]], axis = 1); # As_head.shape = (nx-1,ny-1,nz-2)
    
    ew_tail_indices_x = tf.tile(tf.reshape([self.nx], (-1, 1, 1)), (1, self.ny - 1, self.nz - 2)); # ew_tail_indices_x.shape = (1, ny-1, nz-2)
    ew_head_indices_x = tf.tile(tf.reshape([1], (-1, 1, 1)), (1, self.ny - 1, self.nz - 2)); # ew_head_indices_x,shape = (1, ny-1, nz-2)
    ew_indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (1, 1, self.nz - 2)); # ew_indices_y.shape = (1, ny-1, nz-2)
    ew_indices_z = tf.tile(tf.reshape(tf.range(2, self.nz), (1, 1, -1)), (1, self.ny - 1, 1)); # ew_indices_z.shape = (1, ny-1, nz-2)
    area_east = tf.gather(x, ew_tail_indices_x) / 2 * (tf.gather(y, ew_indices_y + 1) - tf.gather(y, ew_indices_y)) * (tf.gather(z, ew_indices_z + 1) - tf.gather(z, ew_indices_z - 1)) / 2;
    area_west = tf.gather(x, ew_head_indices_x) / 2 * (tf.gather(y, ew_indices_y + 1) - tf.gather(y, ew_indices_y)) * (tf.gather(z, ew_indices_z + 1) - tf.gather(z, ew_indices_z - 1)) / 2;
    flow_east = .5 * self.rho * area_east * (tf.gather_nd(u_old, self.indices(ew_tail_indices_x, ew_indices_y, ew_indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(ew_tail_indices_x, ew_indices_y, ew_indices_z - 1)));
    flow_west = .5 * self.rho * area_west * (tf.gather_nd(u_old, self.indices(ew_head_indices_x, ew_indices_y, ew_indices_z)) + \
                                             tf.gather_nd(u_old, self.indices(ew_head_indices_x, ew_indices_y, ew_indices_z - 1)));
    Ae_tail = tf.math.maximum(-flow_east, 0) + self.mu * area_east / (tf.gather(x, ew_tail_indices_x) - tf.gather(x, ew_tail_indices_x - 1) / 2);
    Aw_head = tf.math.maximum(flow_west, 0) + self.mu * area_west / (tf.gather(x, ew_head_indices_x) - tf.gather(x, ew_head_indices_x - 1) / 2);
    Ae = tf.concat([Ae[:-1,:,:], Ae_tail], axis = 0); # Ae_tail.shape = (nx-1, ny-1, nz-2)
    Aw = tf.concat([Aw_head, Aw[1:,:,:]], axis = 0); # Aw_head.shape = (1, ny-1, nz-2)
    # Calculation
    Apw = (Ae + Aw + An + As + At + Ab) / self.omega_w; # Apw.shape = (nx-1, ny-1, nz-2)
    Apw = tf.pad(Apw, [[1,1],[1,1],[2,1]]); # Apw.shape = (nx+1, ny+1, nz+1)
    # update self.w with iteration
    for i in range(velocity_iter):
      dV = -(tf.gather(y, indices_y - 1) - tf.gather(y, indices_y + 1)) * \
            (tf.gather(z, indices_z - 1) - tf.gather(z, indices_z + 1)) * \
            (2 * tf.gather(x, indices_x) + tf.gather(x, indices_x - 1) + tf.gather(x, indices_x + 1)) * \
            (tf.gather(x, indices_x - 1) - tf.gather(x, indices_x + 1)) / 32;
      dZ = (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
      Valor = Ae * tf.gather_nd(w, self.indices(indices_x + 1, indices_y, indices_z)) + \
              Aw * tf.gather_nd(w, self.indices(indices_x - 1, indices_y, indices_z)) + \
              An * tf.gather_nd(w, self.indices(indices_x, indices_y + 1, indices_z)) + \
              As * tf.gather_nd(w, self.indices(indices_x, indices_y - 1, indices_z)) + \
              At * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z + 1)) + \
              Ab * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z - 1)) - \
              self.beta * (Dcc - Dcu) + \
              dV / dZ * (tf.gather_nd(P, self.indices(indices_x, indices_y, indices_z - 1)) - \
                         tf.gather_nd(P, self.indices(indices_x, indices_y, indices_z)));
      w_update = (1 - self.omega_w) * w_old * tf.pad(Valor, [[1,1],[1,1],[2,1]]) / Apw;
      new_w = tf.where(
                 tf.cast(tf.scatter_nd(
                   tf.reshape(self.indices(indices_x, indices_y, indices_z), (-1, 3)),
                   tf.ones(((self.nx - 1) * (self.ny - 1) * (self.nz - 2),)),
                   (self.nx + 1, self.ny + 1, self.nz + 1)), dtype = tf.bool),
                 tf.scatter_nd(
                   tf.reshape(self.indices(indices_x, indices_y, indices_z), (-1, 3)),
                   tf.reshape(w_update[1:-1,1:-1,2:-1], (-1,)),
                   (self.nx + 1, self.ny + 1, self.nz + 1)),
                 w
               );
      #assert tf.math.reduce_any(tf.math.is_inf(self.w)) != True;
      #assert tf.math.reduce_any(tf.math.is_nan(self.w)) != True;
    return (w, P, u_old, v_old, w_old, x, y, z), (new_w, Apw);
  def pressure(self, pressure_iter):
    u = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    v = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    w = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    P = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    x = tf.placeholder(shape = (self.nx + 1), dtype = self.dtype);
    y = tf.placeholder(shape = (self.ny + 1), dtype = self.dtype);
    z = tf.placeholder(shape = (self.nz + 1), dtype = self.dtype);
    Apu = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    Apv = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    Apw = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    
    indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, self.ny - 1, self.nz - 1)); # indices_x = 1, ..., nx - 1 has totally nx - 1 numbers
    indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 1, 1, self.nz - 1)); # indices_y = 1, ..., ny - 1 has totally ny - 1 numbers
    indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (self.nx - 1, self.ny - 1, 1)); # indices_z = 1, ..., nz - 1 has totally nz - 1 numbers
    area_east = (tf.gather(x, indices_x + 1) + tf.gather(x, indices_x)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_west = (tf.gather(x, indices_x - 1) + tf.gather(x, indices_x)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_north = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x - 1)) / 2 * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_south = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x - 1)) / 2 * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_top = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
               (((tf.gather(x, indices_x + 1) - tf.gather(x, indices_x))/2)**2 - \
                ((tf.gather(x, indices_x) - tf.gather(x, indices_x - 1))/2)**2) / 2;
    area_bottom = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                  (((tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) / 2)**2 - \
                   ((tf.gather(x, indices_x) - tf.gather(x, indices_x - 1)) / 2)**2) / 2;
    Ae = self.rho * area_east**2 / tf.gather_nd(Apu, self.indices(indices_x + 1, indices_y, indices_z));
    Aw = self.rho * area_west**2 / tf.gather_nd(Apu, self.indices(indices_x, indices_y, indices_z));
    An = self.rho * area_north**2 / tf.gather_nd(Apv, self.indices(indices_x, indices_y + 1, indices_z));
    As = self.rho * area_south**2 / tf.gather_nd(Apv, self.indices(indices_x, indices_y, indices_z));
    At = self.rho * area_top**2 / tf.gather_nd(Apw, self.indices(indices_x, indices_y, indices_z + 1));
    Ab = self.rho * area_bottom**2 / tf.gather_nd(Apw, self.indices(indices_x, indices_y, indices_z));
    Ae = tf.concat([Ae[:-1,:,:], tf.zeros_like(Ae[-1:,:,:])], axis = 0); # Ae.shape = (nx-1, ny-1, nz-1)
    Aw = tf.concat([tf.zeros_like(Aw[:1,:,:]), Aw[1:,:,:]], axis = 0); # Aw.shape = (nx-1, ny-1, nz-1)
    An = tf.concat([An[:,:-1,:], tf.zeros_like(An[:,-1:,:])], axis = 1); # An.shape = (nx-1, ny-1, nz-1)
    As = tf.concat([tf.zeros_like(As[:,:1,:]), As[:,1:,:]], axis = 1); # As.shape = (nx-1, ny-1, nz-1)
    At = tf.concat([At[:,:,:-1], tf.zeros_like(At[:,:,-1:])], axis = 2); # At.shape = (nx-1, ny-1, nz-1)
    Ab = tf.concat([tf.zeros_like(Ab[:,:,:1]), Ab[:,:,1:]], axis = 2); # Ab.shape = (nx-1, ny-1, nz-1)
    App = Ae + Aw + An + As + At + Ab;
    App = tf.pad(App, [[1,1],[1,1],[1,1]]);
    App = tf.where(tf.cast(tf.scatter_nd([[1,1,1]], tf.constant([1], dtype = self.dtype), App.shape), dtype = tf.bool), 1e30 * tf.ones_like(App), App);
    App = tf.concat([App[:,:,:-2], 1e30*tf.ones_like(App[:,:,-2:-1]), App[:,:,-1:]], axis = 2);

    area_east = (tf.gather(x, indices_x + 1) + tf.gather(x, indices_x)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_west = (tf.gather(x, indices_x - 1) + tf.gather(x, indices_x)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_north = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x - 1)) / 2 * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_south = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x - 1)) / 2 * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_top = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
               (((tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) / 2)**2 - \
                ((tf.gather(x, indices_x) - tf.gather(x, indices_x - 1)) / 2)**2) / 2;
    area_bottom = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                  (((tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) / 2)**2 - \
                   ((tf.gather(x, indices_x) - tf.gather(x, indices_x - 1)) / 2)**2) / 2;
    Source = self.rho * (area_east * tf.gather_nd(u, self.indices(indices_x + 1, indices_y, indices_z)) - \
                         area_west * tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z))) + \
             self.rho * (area_north * tf.gather_nd(v, self.indices(indices_x, indices_y + 1, indices_z)) - \
                         area_south * tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z))) + \
             self.rho * (area_top * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z + 1)) - \
                         area_bottom * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z)));
    Ae = tf.pad(Ae, [[1,1],[1,1],[1,1]]);
    Aw = tf.pad(Aw, [[1,1],[1,1],[1,1]]);
    An = tf.pad(An, [[1,1],[1,1],[1,1]]);
    As = tf.pad(As, [[1,1],[1,1],[1,1]]);
    At = tf.pad(At, [[1,1],[1,1],[1,1]]);
    Ab = tf.pad(Ab, [[1,1],[1,1],[1,1]]);
    Source = tf.pad(Source, [[1,1],[1,1],[1,1]]);
    Pp = tf.zeros((self.nx - 1, self.ny - 1, self.nz - 1), dtype = self.dtype);
    for i in range(pressure_iter):
      padded_Pp = tf.pad(Pp, [[1,1],[1,1],[1,1]]);
      Pp = Pp + \
           self.omega_pp / tf.gather_nd(App, self.indices(indices_x, indices_y, indices_z)) * \
           (tf.gather_nd(Ae, self.indices(indices_x, indices_y, indices_z)) * tf.gather_nd(padded_Pp, self.indices(indices_x + 1, indices_y, indices_z)) + \
            tf.gather_nd(Aw, self.indices(indices_x, indices_y, indices_z)) * tf.gather_nd(padded_Pp, self.indices(indices_x - 1, indices_y, indices_z)) + \
            tf.gather_nd(An, self.indices(indices_x, indices_y, indices_z)) * tf.gather_nd(padded_Pp, self.indices(indices_x, indices_y + 1, indices_z)) + \
            tf.gather_nd(As, self.indices(indices_x, indices_y, indices_z)) * tf.gather_nd(padded_Pp, self.indices(indices_x, indices_y - 1, indices_z)) + \
            tf.gather_nd(At, self.indices(indices_x, indices_y, indices_z)) * tf.gather_nd(padded_Pp, self.indices(indices_x, indices_y, indices_z + 1)) + \
            tf.gather_nd(Ab, self.indices(indices_x, indices_y, indices_z)) * tf.gather_nd(padded_Pp, self.indices(indices_x, indices_y, indices_z - 1)) - \
            tf.gather_nd(Source, self.indices(indices_x, indices_y, indices_z)) - \
            tf.gather_nd(padded_Pp, self.indices(indices_x, indices_y, indices_z)) * \
            tf.gather_nd(App, self.indices(indices_x, indices_y, indices_z)));
      #assert tf.math.reduce_any(tf.math.is_inf(Pp)) != True;
      #assert tf.math.reduce_any(tf.math.is_nan(Pp)) != True;
    Pp = tf.pad(Pp, [[1,1],[1,1],[1,1]]);
    new_P = P + self.omega_p * Pp;
    #assert tf.math.reduce_any(tf.math.is_inf(self.P)) != True;
    #assert tf.math.reduce_any(tf.math.is_nan(self.P)) != True;
    return (u, v, w, P, x, y, z, Apu, Apv, Apw), (new_P, Pp);
  def ensure_quality(self,):
    u = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    v = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    w = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    x = tf.placeholder(shape = (self.nx + 1), dtype = self.dtype);
    y = tf.placeholder(shape = (self.ny + 1), dtype = self.dtype);
    z = tf.placeholder(shape = (self.nz + 1), dtype = self.dtype);
    Pp = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    Apu = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    Apv = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    Apw = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);

    # correcting u
    u_indices_x = tf.tile(tf.reshape(tf.range(2, self.nx), (-1, 1, 1)), (1, self.ny - 1, self.nz - 1)); # u_indices.x.shape = (nx-2, ny-1, nz-1)
    u_indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 2, 1, self.nz - 1)); # u_indices_y.shape = (nx-2, ny-1, nz-1)
    u_indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (self.nx - 2, self.ny - 1, 1)); # u_indices_z.shape = (nx-2, ny-1, nz-1)
    area_east = (tf.gather(x, u_indices_x + 1) + tf.gather(x, u_indices_x)) / 2 * \
                (tf.gather(y, u_indices_y + 1) - tf.gather(y, u_indices_y - 1)) / 2 * \
                (tf.gather(z, u_indices_z + 1) - tf.gather(z, u_indices_z - 1)) / 2;
    area_west = (tf.gather(x, u_indices_x - 1) + tf.gather(x, u_indices_x)) / 2 * \
                (tf.gather(y, u_indices_y + 1) - tf.gather(y, u_indices_y - 1)) / 2 * \
                (tf.gather(z, u_indices_z + 1) - tf.gather(z, u_indices_z - 1)) / 2;
    u_update = tf.gather_nd(u, self.indices(u_indices_x, u_indices_y, u_indices_z)) + \
               1 / tf.gather_nd(Apu, self.indices(u_indices_x, u_indices_y, u_indices_z)) * \
               (area_west * tf.gather_nd(Pp, self.indices(u_indices_x - 1, u_indices_y, u_indices_z)) - \
                area_east * tf.gather_nd(Pp, self.indices(u_indices_x, u_indices_y, u_indices_z)));
    new_u = tf.where(
               tf.cast(tf.scatter_nd(
                 tf.reshape(self.indices(u_indices_x, u_indices_y, u_indices_z), (-1, 3)),
                 tf.ones(((self.nx - 2) * (self.ny - 1) * (self.nz - 1),)),
                 (self.nx + 1, self.ny + 1, self.nz + 1)), dtype = tf.bool),
               tf.scatter_nd(
                 tf.reshape(self.indices(u_indices_x, u_indices_y, u_indices_z), (-1, 3)),
                 tf.reshape(u_update, (-1,)),
                 (self.nx + 1, self.ny + 1, self.nz + 1)),
               u
             );
    # correcting v
    v_indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, self.ny - 2, self.nz - 1)); # v_indices_x.shape = (nx-1, ny-2, nz-1)
    v_indices_y = tf.tile(tf.reshape(tf.range(2, self.ny), (1, -1, 1)), (self.nx - 1, 1, self.nz - 1)); # v_indices_y.shape = (nx-1, ny-2, nz-1)
    v_indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (self.nx - 1, self.ny - 2, 1)); # v_indices_z.shape = (nx-1, ny-2, nz-1)
    area_north = (tf.gather(x, v_indices_x + 1) - tf.gather(x, v_indices_x - 1)) / 2 * \
                 (tf.gather(z, v_indices_z + 1) - tf.gather(z, v_indices_z - 1)) / 2;
    area_south = (tf.gather(x, v_indices_x + 1) - tf.gather(x, v_indices_x - 1)) / 2 * \
                 (tf.gather(z, v_indices_z + 1) - tf.gather(z, v_indices_z - 1)) / 2;
    v_update = tf.gather_nd(v, self.indices(v_indices_x, v_indices_y, v_indices_z)) + \
               1 / tf.gather_nd(Apv, self.indices(v_indices_x, v_indices_y, v_indices_z)) * \
               (area_south * tf.gather_nd(Pp, self.indices(v_indices_x, v_indices_y - 1, v_indices_z)) - \
                area_north * tf.gather_nd(Pp, self.indices(v_indices_x, v_indices_y, v_indices_z)));
    new_v = tf.where(
               tf.cast(tf.scatter_nd(
                 tf.reshape(self.indices(v_indices_x, v_indices_y, v_indices_z), (-1, 3)),
                 tf.ones(((self.nx - 1) * (self.ny - 2) * (self.nz - 1),)),
                 (self.nx + 1, self.ny + 1, self.nz + 1)), dtype = tf.bool),
               tf.scatter_nd(
                 tf.reshape(self.indices(v_indices_x, v_indices_y, v_indices_z), (-1, 3)),
                 tf.reshape(v_update, (-1,)),
                 (self.nx + 1, self.ny + 1, self.nz + 1)),
               v
             );
    # correcting w
    w_indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, self.ny - 1, self.nz - 2)); # w_indices_x.shape = (nx-1, ny-1, nz-2)
    w_indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 1, 1, self.nz - 2)); # w_indices_y.shape = (nx-1, ny-1, nz-2)
    w_indices_z = tf.tile(tf.reshape(tf.range(2, self.nz), (1, 1, -1)), (self.nx - 1, self.ny - 1, 1)); # w_indices_z.shape = (nx-1, ny-1, nz-2)
    area_top = (tf.gather(y, w_indices_y + 1) - tf.gather(y, w_indices_y - 1)) / 2 * \
               (((tf.gather(x, w_indices_x + 1) - tf.gather(x, w_indices_x)) / 2)**2 - \
                ((tf.gather(x, w_indices_x) - tf.gather(x, w_indices_x - 1)) / 2)**2) / 2;
    area_bottom = (tf.gather(y, w_indices_y + 1) - tf.gather(y, w_indices_y - 1)) / 2 * \
                  (((tf.gather(x, w_indices_x + 1) - tf.gather(x, w_indices_x)) / 2)**2 - \
                   ((tf.gather(x, w_indices_x) - tf.gather(x, w_indices_x - 1)) / 2)**2) / 2;
    w_update = tf.gather_nd(w, self.indices(w_indices_x, w_indices_y, w_indices_z)) + \
               1 / tf.gather_nd(Apw, self.indices(w_indices_x, w_indices_y, w_indices_z)) * \
               (area_bottom * tf.gather_nd(Pp, self.indices(w_indices_x, w_indices_y, w_indices_z - 1)) - \
                area_top * tf.gather_nd(Pp, self.indices(w_indices_x, w_indices_y, w_indices_z)));
    new_w = tf.where(
               tf.cast(tf.scatter_nd(
                 tf.reshape(self.indices(w_indices_x, w_indices_y, w_indices_z), (-1, 3)),
                 tf.ones(((self.nx - 1) * (self.ny - 1) * (self.nz - 2),)),
                 (self.nx + 1, self.ny + 1, self.nz + 1)), dtype = tf.bool),
               tf.scatter_nd(
                 tf.reshape(self.indices(w_indices_x, w_indices_y, w_indices_z), (-1, 3)),
                 tf.reshape(w_update, (-1,)),
                 (self.nx + 1, self.ny + 1, self.nz + 1)),
               w
             );
    return (u, v, w, x, y, z, Pp, Apu, Apv, Apw), (new_u, new_v, new_w);
  def error_source(self, errors):
    u = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    v = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    w = tf.placeholder(shape = (self.nx + 1, self.ny + 1, self.nz + 1), dtype = self.dtype);
    x = tf.placeholder(shape = (self.nx + 1), dtype = self.dtype);
    y = tf.placeholder(shape = (self.ny + 1), dtype = self.dtype);
    z = tf.placeholder(shape = (self.nz + 1), dtype = self.dtype);

    indices_x = tf.tile(tf.reshape(tf.range(1, self.nx), (-1, 1, 1)), (1, self.ny - 1, self.nz - 1)); # indices_x.shape = (nx-1, ny-1, nz-1)
    indices_y = tf.tile(tf.reshape(tf.range(1, self.ny), (1, -1, 1)), (self.nx - 1, 1, self.nz - 1)); # indices_y.shape = (nx-1, ny-1, nz-1)
    indices_z = tf.tile(tf.reshape(tf.range(1, self.nz), (1, 1, -1)), (self.nx - 1, self.ny - 1, 1)); # indices_z.shape = (nx-1, ny-1, nz-1)
    area_east = (tf.gather(x, indices_x + 1) + tf.gather(x, indices_x)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_west = (tf.gather(x, indices_x - 1) + tf.gather(x, indices_x)) / 2 * \
                (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_north = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x - 1)) / 2 * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_south = (tf.gather(x, indices_x + 1) - tf.gather(x, indices_x - 1)) / 2 * \
                 (tf.gather(z, indices_z + 1) - tf.gather(z, indices_z - 1)) / 2;
    area_top = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
               (((tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) / 2)**2 - \
                ((tf.gather(x, indices_x) - tf.gather(x, indices_x - 1)) / 2)**2) / 2;
    area_bottom = (tf.gather(y, indices_y + 1) - tf.gather(y, indices_y - 1)) / 2 * \
                  (((tf.gather(x, indices_x + 1) - tf.gather(x, indices_x)) / 2)**2 - \
                   ((tf.gather(x, indices_x) - tf.gather(x, indices_x - 1)) / 2)**2) / 2;
    Source = self.rho * (area_east * tf.gather_nd(u, self.indices(indices_x + 1, indices_y, indices_z)) - \
                         area_west * tf.gather_nd(u, self.indices(indices_x, indices_y, indices_z))) + \
             self.rho * (area_north * tf.gather_nd(v, self.indices(indices_x, indices_y + 1, indices_z)) - \
                         area_south * tf.gather_nd(v, self.indices(indices_x, indices_y, indices_z))) + \
             self.rho * (area_top * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z + 1)) - \
                         area_bottom * tf.gather_nd(w, self.indices(indices_x, indices_y, indices_z)));
    error = tf.math.sqrt(tf.math.reduce_sum(Source**2));
    return (u, v, w, x, y, z), (error,);
  def solve(self, iteration = 10, velocity_iter = 10, pressure_iter = 20):
    errors = list();
    def debug(i):
      u_is_nan = 'true' if tf.math.reduce_any(tf.math.is_nan(self.u)) else 'false';
      v_is_nan = 'true' if tf.math.reduce_any(tf.math.is_nan(self.v)) else 'false';
      w_is_nan = 'true' if tf.math.reduce_any(tf.math.is_nan(self.w)) else 'false';
      p_is_nan = 'true' if tf.math.reduce_any(tf.math.is_nan(self.P)) else 'false';
      print('step: %d u is nan: %s, v is nan: %s, w is nan: %s, p is nan: %s' % (i, u_is_nan, v_is_nan, w_is_nan, p_is_nan));
      u_is_inf = 'true' if tf.math.reduce_any(tf.math.is_inf(self.u)) else 'false';
      v_is_inf = 'true' if tf.math.reduce_any(tf.math.is_inf(self.v)) else 'false';
      w_is_inf = 'true' if tf.math.reduce_any(tf.math.is_inf(self.w)) else 'false';
      p_is_inf = 'true' if tf.math.reduce_any(tf.math.is_inf(self.P)) else 'false';
      print('step: %d u is inf: %s, v is inf: %s, w is inf: %s, p is inf: %s' % (i, u_is_inf, v_is_inf, w_is_inf, p_is_inf));
    for i in range(iteration):
      debug(i);
      u_old, v_old, w_old = tf.identity(self.u), tf.identity(self.v), tf.identity(self.w);
      Apu = self.momento_x(u_old, v_old, w_old, velocity_iter);
      debug(i);
      Apv = self.momento_y(u_old, v_old, w_old, velocity_iter);
      debug(i);
      Apw = self.momento_z(u_old, v_old, w_old, velocity_iter);
      debug(i);
      self.set_conditions();
      Pp = self.pressure(Apu, Apv, Apw, pressure_iter);
      self.set_conditions();
      self.ensure_quality(Pp, Apu, Apv, Apw);
      self.set_conditions();
      self.error_source(errors);
      if tf.math.is_nan(errors[-1]): break;
      if i > 1 and tf.math.abs(errors[-2] - errors[-1]) / tf.math.abs(errors[-2]) < 1e-12: break;
      if errors[-1] / errors[0] < 1e-3: break;
    return self.u, self.v, self.w, self.P;

if __name__ == "__main__":
  simple = SIMPLE();
  u, v, w, P = simple.solve();
  print(u,v,w,P);

import numpy as np
from envs.diffdart_env import DiffDartEnv
from pdb import set_trace as bp
import torch
from .utils import ComputeCostGrad


class DartSnake7LinkEnv(DiffDartEnv):#, utils.EzPickle):
    def __init__(self):
        #self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        #self.action_scale = 200
        #self.include_action_in_obs = False
        #self.randomize_dynamics = False
        #obs_dim = 17

        #if self.include_action_in_obs:
        #    obs_dim += len(self.control_bounds[0])
        #    self.prev_a = np.zeros(len(self.control_bounds[0]))
        frame_skip = 1
        DiffDartEnv.__init__(self, 'snake_7link.skel', frame_skip)#, obs_dim, self.control_bounds, disableViewer=True)
        self.ndofs = self.robot_skeleton.getNumDofs()
        self.control_dofs = np.arange(2,self.ndofs) 

#        if self.randomize_dynamics:
#            self.bodynode_original_masses = []
#            self.bodynode_original_frictions = []
#            for bn in self.robot_skeleton.bodynodes:
#                self.bodynode_original_masses.append(bn.mass())
#                self.bodynode_original_frictions.append(bn.friction_coeff())
#
#        self.dart_world.set_collision_detector(3)

#        for i in range(0, len(self.robot_skeleton.bodynodes)):
#            self.robot_skeleton.bodynodes[i].set_friction_coeff(0)
#        self.robot_skeleton.bodynodes[-1].set_friction_coeff(0)

#        utils.EzPickle.__init__(self)
    
    def running_cost(self, x, u, compute_grads = False): #define the running cost
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)
        #bp()
        mask = torch.zeros(self.ndofs*2)
        mask[self.ndofs] = 1.0
        mask[self.ndofs+2]=0.001
        x_target = torch.zeros(self.ndofs*2)
        x_target[self.ndofs] = 10.0



        #---------------------------Enter running cost:-----------------------------------------------------------
        run_cost = torch.sum(1e-3*torch.mul(u,u)) #example of quadratic cost
        run_cost += torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = (v0-10)^2+0.001 v2^2 : make v0 "big", i.e., close to 10, while keeping v2 low
        #---------------------------------------------------------------------------------------------------------
        #bp()
        #Autodiff gradient and Hessian calculation
        if compute_grads:
            run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu = ComputeCostGrad(run_cost, x, u=u)
            return run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu
        else:
            return run_cost.detach().numpy()

    def terminal_cost(self, x, compute_grads = False): #define the terminal cost
        x = torch.tensor(x, requires_grad=True)

        #---------------------------Enter terminal cost:-----------------------------------------------------------
        #x_target = torch.FloatTensor([0., 0., 0., 0.])
        #coeff = torch.FloatTensor([0.0, 1000., 60., 100.])
        #ter_cost = torch.sum(torch.mul(coeff,torch.mul(x-x_target,x-x_target))) #example cT*(x-x_target)*2
        mask = torch.zeros(self.ndofs*2)
        mask[0] = 10.0
        mask[2]=0.1
        x_target = torch.zeros(self.ndofs*2)
        x_target[0] = 10.0
        ter_cost = torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = 10(x0-10)^2+0.1 x2^2 : make x0 "big", i.e., close to 10, while keeping x2 low
        #--------------------------------------------------------------------------------------------------------- 

        #Autodiff gradient and Hessian calculation
        if compute_grads:
            ter_cost, grad_x, Hess_xx = ComputeCostGrad(ter_cost, x)
            
            return ter_cost, grad_x, Hess_xx
        else:
            return ter_cost.detach().numpy()

    def run_cost(self):
        return self.running_cost
    def ter_cost(self):
        return self.terminal_cost









#-----------------------------------------------------------------------------------------------------
#------ WARNING: THE FOLLOWING ARE NOT USED. ONLY RETAINED IN CASE WE WANT TO TRAIN AN RL POLICY -----
#-----------------------------------------------------------------------------------------------------






    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            for bn in self.robot_skeleton.bodynodes:
                bn_vel = bn.com_spatial_velocity()
                norm_dir = bn.to_world([0, 0, 1]) - bn.to_world([0, 0, 0])
                vel_pos = bn_vel[3:] + np.cross(bn_vel[0:3], norm_dir) * 0.05
                vel_neg = bn_vel[3:] - np.cross(bn_vel[0:3], norm_dir) * 0.05
                fluid_force = [0.0, 0.0, 0.0]
                if np.dot(vel_pos, norm_dir) > 0.0:
                    fluid_force = -50.0 * np.dot(vel_pos, norm_dir) * norm_dir
                if np.dot(vel_neg, norm_dir) < 0.0:
                    fluid_force = -50.0 * np.dot(vel_neg, norm_dir) * norm_dir
                bn.add_ext_force(fluid_force)

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        if self.include_action_in_obs:
            self.prev_a = np.copy(clamped_control)

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter = self.robot_skeleton.q[0]
        deviation = self.robot_skeleton.q[2]

        alive_bonus = 0.1
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= np.abs(deviation) * 0.1
        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and abs(deviation) < 1.5)
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_action_in_obs:
            state = np.concatenate([state, self.prev_a])

        return state


    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        if self.include_action_in_obs:
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        self.accumulated_rew = 0.0
        self.num_steps = 0.0

        if self.randomize_dynamics:
            for i in range(len(self.robot_skeleton.bodynodes)):
                self.robot_skeleton.bodynodes[i].set_mass(
                    self.bodynode_original_masses[i] + np.random.uniform(-1.5, 1.5))
                self.robot_skeleton.bodynodes[i].set_friction_coeff(
                    self.bodynode_original_frictions[i] + np.random.uniform(-0.5, 0.5))

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
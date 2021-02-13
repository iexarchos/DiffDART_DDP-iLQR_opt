import numpy as np
from envs.diffdart_env import DiffDartEnv
from pdb import set_trace as bp
import torch
from .utils import ComputeCostGrad

class DartHopperEnv(DiffDartEnv):
    def __init__(self,FD=False):
        #self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        #self.action_scale = 200
        #obs_dim = 11
        frame_skip = 1
        DiffDartEnv.__init__(self, 'hopper_capsule.skel', frame_skip, dt=0.01,FD=FD)#, obs_dim, self.control_bounds, disableViewer=True)
        self.ndofs = self.robot_skeleton.getNumDofs()
        #bp()
        self.control_dofs = np.arange(0,self.ndofs) #TODO needs to change!
        #try:
        #    self.dart_world.set_collision_detector(3)
        #except Exception as e:
        #    print('Does not have ODE collision detector, reverted to bullet collision detector')
        #    self.dart_world.set_collision_detector(2)
        

        #utils.EzPickle.__init__(self)


    def running_cost(self, x, u, compute_grads = False): #define the running cost
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)
        #bp()
        mask = torch.zeros(self.ndofs*2)
        mask[self.ndofs] = 1.0
        #mask[self.ndofs+2]=0.001
        x_target = torch.zeros(self.ndofs*2)
        x_target[self.ndofs] = 1.0



        #---------------------------Enter running cost:-----------------------------------------------------------
        run_cost = torch.sum(1e-3*torch.mul(u,u)) #example of quadratic cost
        run_cost += torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = (v0-10)^2: make v0 "big", i.e., close to 10
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
        #mask[2]=0.1
        x_target = torch.zeros(self.ndofs*2)
        x_target[0] = 1.0
        ter_cost = torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = 10(x0-10)^2+0.1 x2^2 : make x0 "big", i.e., close to 10
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










    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]


        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        # uncomment the line below to enable joint limit penalty, which helps learning
        reward -= 5e-1 * joint_limit_penalty

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (height < 1.8) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

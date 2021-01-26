import numpy as np
from envs.diffdart_env import DiffDartEnv
from pdb import set_trace as bp
import torch
from .utils import ComputeCostGrad

class DartDogEnv(DiffDartEnv):
    def __init__(self,FD=False):
        #self.control_bounds = np.array([[1.0]*16,[-1.0]*16])
        #self.action_scale = 200
        #obs_dim = 43
        frame_skip=1
        DiffDartEnv.__init__(self, 'dog.skel', frame_skip,FD=FD)#, obs_dim, self.control_bounds, disableViewer=False)
        self.ndofs = self.robot_skeleton.getNumDofs()
        #bp()
        self.control_dofs = np.arange(0,self.ndofs)
        #utils.EzPickle.__init__(self)


    def running_cost(self, x, u, compute_grads = False): #define the running cost
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)
        #bp()
        mask = torch.zeros(self.ndofs*2)
        mask[self.ndofs] = 1.0
        #mask[self.ndofs+2]=0.001
        x_target = torch.zeros(self.ndofs*2)
        x_target[self.ndofs] = 10.0



        #---------------------------Enter running cost:-----------------------------------------------------------
        run_cost = torch.sum(1e-3*torch.mul(u,u)) #example of quadratic cost
        run_cost += torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = (v0-10)^2: make v0 "big", i.e., close to x_target
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
        x_target[0] = 10.0
        ter_cost = torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = 10(x0-10)^2+0.1 x2^2 : make x0 "big", i.e., close to xtarget
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


    def step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.do_simulation(tau, self.frame_skip)
        posafter = self.robot_skeleton.bodynodes[0].com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        side_deviation = abs(self.robot_skeleton.bodynodes[0].com()[2])

        alive_bonus = 1.0
        reward = 0.6*(posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (height < 1.8) and (side_deviation < .4))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -15.5
        self.track_skeleton_id = 0

import numpy as np
from envs.diffdart_env import DiffDartEnv
from pdb import set_trace as bp
import torch
from .utils import ComputeCostGrad

# swing up and balance of double inverted pendulum
class DartDoubleInvertedPendulumEnv(DiffDartEnv):
    def __init__(self):
        control_bounds = np.array([[1.0],[-1.0]])
        #self.action_scale = 40
        frame_skip = 1
        DiffDartEnv.__init__(
            self, 'inverted_double_pendulum.skel', frame_skip, dt=0.02)#, 8, control_bounds, dt=0.01)

        #self.init_qpos = np.array(self.robot_skeleton.q).copy()
        #self.init_qvel = np.array(self.robot_skeleton.dq).copy()




    def running_cost(self, x, u, compute_grads = False): #define the running cost
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)

        #---------------------------Enter running cost:-----------------------------------------------------------
        run_cost = torch.sum(0.005*torch.mul(u,u)) #example of quadratic cost
        #---------------------------------------------------------------------------------------------------------

        #Autodiff gradient and Hessian calculation
        if compute_grads:
            run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu = ComputeCostGrad(run_cost, x, u=u)
            return run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu
        else:
            return run_cost.detach().numpy()

    def terminal_cost(self, x, compute_grads = False): #define the terminal cost
        x = torch.tensor(x, requires_grad=True)

        #---------------------------Enter terminal cost:-----------------------------------------------------------
        x_target = torch.FloatTensor([0., 0., 0., 0., 0. , 0.])
        coeff = torch.FloatTensor([0.0, 100., 100., 60., 10., 10.])
        ter_cost = torch.sum(torch.mul(coeff,torch.mul(x-x_target,x-x_target))) #example cT*(x-x_target)*2
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

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        base = self.robot_skeleton.name_to_body['cart'].to_world()[1]
        weight_sz = 0.02 # TODO: Make sure this doesn't change.
        max_height = 0.6
        raw_height = self.robot_skeleton.name_to_body['weight'].to_world()[1]
        # Have the same scaling as the gym env.
        height = 2.0*(raw_height - base - weight_sz)/max_height

        v1, v2 = self.robot_skeleton.dq[1:3]

        alive_bonus = 10.
        dist_penalty = 0.01*ob[0]**2 + (height - 2.)**2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        reward = alive_bonus - dist_penalty - vel_penalty

        done = bool(height <= 1)
        return ob, reward, done, {}


    def _get_obs(self):
        return np.concatenate([
            self.robot_skeleton.q[:1],
            np.sin(self.robot_skeleton.q[1:]),
            np.cos(self.robot_skeleton.q[1:]),
            self.robot_skeleton.dq
        ]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.init_qpos + \
               self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        qvel = self.init_qvel + \
               self.np_random.randn(self.robot_skeleton.ndofs) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
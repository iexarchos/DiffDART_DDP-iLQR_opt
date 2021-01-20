import numpy as np
import torch
import diffdart as dart
from pdb import set_trace as bp
import time


class DDP_Traj_Optimizer():
	def __init__(self,
				 Env, #diffDart Env
				 T,     # Time horizon in seconds
				 X0 = None, # State initial condition (optional; if not provided, it is obtained from world)
				 U_guess = None, # Control initial guess (optional)
				 lr = 0.5 # Learning rate
				 ):
		#From the diffDart environment, obtain the world object and the running and terminal cost functions
		self.Env = Env()
		self.world = self.Env.dart_world
		self.robot = self.Env.robot_skeleton
		self.running_cost = self.Env.run_cost()
		self.terminal_cost = self.Env.ter_cost()
		self.T = T
		self.dt = self.Env.dt #this includes frameskip!
		self.N = int(T / self.dt) #number of timesteps
		self.dofs =  self.robot.getNumDofs()
		self.n_states = self.dofs*2
		self.n_controls = self.dofs # change that as soon as you can specify control_DoFs separately.
		self.lr = lr
		self.frame_skip = self.Env.frame_skip
		assert self.frame_skip == 1 #DDP currently supports no frame skip!
		
		#----- Initializing numpy arrays------
		# Initialize state and control traj
		self.X = np.zeros((self.N,self.n_states))
		if X0 is not None:
			self.X[0,:] = X0
		else:
			self.X0 = np.concatenate((self.robot.getPositions(), self.robot.getVelocities()))

		self.robot.setPositions(self.X[0,0:self.dofs])
		self.robot.setVelocities(self.X[0,self.dofs:])
		self.U = np.zeros((self.N-1,self.n_controls))
		if U_guess is not None:
			self.U = U_guess.copy()

		#initialize traj corrections:
		self.dx = np.zeros((self.N,self.n_states))
		self.du = np.zeros((self.N-1,self.n_controls))

		# initialize Cost derivative matrices
		self.L0 = np.zeros((self.N-1,))
		self.Lx = np.zeros((self.N-1,self.n_states))
		self.Lu = np.zeros((self.N-1,self.n_controls))
		self.Lxx = np.zeros((self.N-1,self.n_states,self.n_states))
		self.Luu = np.zeros((self.N-1,self.n_controls,self.n_controls))
		self.Lux = np.zeros((self.N-1,self.n_states,self.n_controls))
		self.Lxu = np.zeros((self.N-1,self.n_controls,self.n_states))

		# initialize Jacobians of dynamics, w.r.t. state x and control u
		self.Fx = np.zeros((self.N-1,self.n_states,self.n_states))   
		self.Fu = np.zeros((self.N-1,self.n_states,self.n_controls)) 

		# initialize Value function and derivatives
		self.V = np.zeros((self.N,)) 
		self.Vx = np.zeros((self.N,self.n_states))
		self.Vxx = np.zeros((self.N,self.n_states,self.n_states))
		

		# initialize trajectory and control update matrices
		self.L_k = np.zeros((self.N-1,self.n_controls,self.n_states))
		self.l_k = np.zeros((self.N-1,self.n_controls))


		self.Cost = []

	def optimize(self,maxIter,thresh=None):
		t = time.time()
		prev_cost = np.inf
		for i in range(maxIter):
			self.forward_pass()
			self.backward_pass()
			self.update_control()
			curr_cost = self.trajectory_rollout()
			self.Cost.append(curr_cost)
			print('Iteration: ', i+1, ', trajectory cost: ', curr_cost)
			if thresh is not None:
				if abs(prev_cost-curr_cost)<thresh:
					print('Optimization threshold met, exiting...')
					break
				else:
					prev_cost = curr_cost.copy()
		print('---Optimization completed in ',time.time()-t,'sec---')
		return self.X, self.U, self.Cost


	def forward_pass(self):
		for j in range(self.N-1):
        
			l0, l_x, l_xx, l_u, l_uu, l_ux, l_xu = self.running_cost(self.X[j,:],self.U[j,:],compute_grads=True)
			self.L0[j] = l0*self.dt
			self.Lx[j,:] = l_x*self.dt
			self.Lu[j,:] = l_u*self.dt
			self.Lxx[j,:] = l_xx*self.dt
			self.Luu[j,:,:] = l_uu*self.dt
			self.Lux[j,:,:] = l_ux*self.dt
			self.Lxu[j,:,:] = l_xu*self.dt

			x_next, Fx, Fu = self.dynamics(self.X[j,:],self.U[j,:],compute_grads=True)

			self.X[j+1,:] = x_next
			self.Fx[j,:,:] = Fx
			self.Fu[j,:,:] = Fu    
		
	def backward_pass(self):
		# initialize backward pass:
		self.V[-1], self.Vx[-1,:], self.Vxx[-1,:,:] = self.terminal_cost(self.X[-1,:],compute_grads=True)
		for j in range(self.N-2,-1,-1):
			Q0 = self.L0[j] +self.V[j+1]
			Qx = self.Lx[j,:] + self.Fx[j,:,:].T.dot(self.Vx[j+1,:])
			Qu = self.Lu[j,:] + self.Fu[j,:,:].T.dot(self.Vx[j+1,:])
			Qxx = self.Lxx[j,:,:] + self.Fx[j,:,:].T.dot(self.Vxx[j+1,:,:].dot(self.Fx[j,:,:]))
			Qxu = self.Lxu[j,:,:] + self.Fx[j,:,:].T.dot(self.Vxx[j+1,:,:].dot(self.Fu[j,:,:])).T
			Qux = self.Lux[j,:,:] + self.Fu[j,:,:].T.dot(self.Vxx[j+1,:,:].dot(self.Fx[j,:,:])).T
			Quu = self.Luu[j,:,:] + self.Fu[j,:,:].T.dot(self.Vxx[j+1,:,:].dot(self.Fu[j,:,:]))

			if Quu.shape[0] == 1:
				if Quu == 0:
					Quu+=1e-5
					print('Warning: singular Quu')
				Quu_inv = 1.0/Quu
			else:
				if not self.is_invertible(Quu):
					Quu+=1e-5*np.eye(Quu.shape[0])
					#bp()
					print('Warning: singular Quu')
				Quu_inv = np.linalg.inv(Quu)

			self.L_k[j,:,:] = -Quu_inv.dot(Qux.T) 
			self.l_k[j,:] = -Quu_inv.dot(Qu)

			self.V[j] = Q0 + Qu.T.dot(self.l_k[j,:])+0.5*self.l_k[j,:].T.dot(Quu.dot(self.l_k[j,:]))
			self.Vx[j,:] = Qx + self.L_k[j,:,:].T.dot(Qu)+self.l_k[j,:].dot(Qxu)+self.L_k[j,:,:].T.dot(Quu.dot(self.l_k[j,:]))
			self.Vxx[j,:,:] = Qxx + self.L_k[j,:,:].T.dot(Qxu) + Qux.dot(self.L_k[j,:,:]) + self.L_k[j,:,:].T.dot(Quu.dot(self.L_k[j,:,:]))

	def update_control(self):
		for j in range(self.N-1):
			self.du[j,:] = self.l_k[j,:] + self.L_k[j,:,:].dot(self.dx[j,:])
			self.dx[j+1,:] = self.Fx[j,:,:].dot(self.dx[j,:])+self.Fu[j,:,:].dot(self.du[j,:])
			self.U[j,:] = self.U[j,:] + self.lr *self.du[j,:]

	def trajectory_rollout(self,render=False):
		cost = 0
		for j in range(self.N-1):
			self.X[j+1,:] = self.dynamics(self.X[j,:], self.U[j,:])
			cost += self.running_cost(self.X[j,:],self.U[j,:])*self.dt
		cost += self.terminal_cost(self.X[-1,:])
		return cost

	def dynamics(self,x,u,compute_grads = False):
		pos = x[:self.dofs]
		vel = x[self.dofs:]
		self.robot.setPositions(pos)
		self.robot.setVelocities(vel)
		

		for _ in range(self.frame_skip):
			self.robot.setForces(u)
			snapshot = dart.neural.forwardPass(self.world)
		x_next = np.concatenate((self.robot.getPositions(), self.robot.getVelocities()))
		if compute_grads:
			forceVel = snapshot.getForceVelJacobian(self.world, perfLog=None)
			forcePos = np.zeros_like(forceVel)
			velVel = snapshot.getVelVelJacobian(self.world)
			velPos = snapshot.getVelPosJacobian(self.world)
			posVel = snapshot.getPosVelJacobian(self.world)
			posPos = snapshot.getPosPosJacobian(self.world)

			Fx = np.block([
			    [posPos[-self.dofs:,-self.dofs:], velPos[-self.dofs:,-self.dofs:]],
			    [posVel[-self.dofs:,-self.dofs:], velVel[-self.dofs:,-self.dofs:]]
			])

			Fu = np.block([
			    [forcePos[-self.dofs:,-self.dofs:]],
			    [forceVel[-self.dofs:,-self.dofs:]]
			])			
			return x_next, Fx, Fu
		else:
			return x_next

	def is_invertible(self,M):
		return M.shape[0]==M.shape[1] and np.linalg.matrix_rank(M) == M.shape[0]

	def simulate_traj(self, X, U, render = False):
		cost = 0
		if render:
			self.gui = dart.DartGUI()
			self.gui.serve(8080)
			self.gui.stateMachine().renderWorld(self.world)
			input('Press enter to begin rendering')
		for j in range(self.N-1):
			X[j+1,:] = self.dynamics(X[j,:], U[j,:])
			if render:
				self.gui.stateMachine().renderWorld(self.world)
				time.sleep(self.dt)
			cost += self.running_cost(X[j,:],U[j,:])*self.dt
		cost += self.terminal_cost(self.X[-1,:])
		return cost


	
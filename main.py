import numpy as np
from pdb import set_trace as bp
from DDP_opt import DDP_Traj_Optimizer
from envs.cart_pole import DartCartPoleEnv
from envs.snake_7link import DartSnake7LinkEnv
from envs.reacher2d import DartReacher2dEnv
from envs.half_cheetah import DartHalfCheetahEnv
from envs.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
import diffdart as dart


def main():
	#Choose example and corresponding initial condition X0:

	#Env = DartCartPoleEnv
	#X0 = [0., 3.14, 0., 0.] 

	#Env = DartSnake7LinkEnv
	#X0 = None

	#Env = DartReacher2dEnv
	#X0 = None

	#Env = DartDoubleInvertedPendulumEnv
	#X0 = [0., 3.14, 0.0, 0., 0., 0.] 

	Env = DartHalfCheetahEnv
	X0 = None

	

	T = 2.0 # planning horizon in seconds
	lr = 0.1 #learning rate

	maxIter = 40 # maximum number of iterations
	threshold = 0.001 # Optional, set to 'None' otherwise. Early stopping of optimization if cost doesn't improve more than this between iterations.

	DDP = DDP_Traj_Optimizer(Env=Env,T=T,X0=X0, U_guess='random')
	x,u,cost = DDP.optimize(maxIter = maxIter, thresh=threshold, lr=lr)

	bp()
	c = DDP.simulate_traj(x, u, render = True)
	print('Optimal trajectory cost: ', c)
	from matplotlib import pyplot as plt
	plt.figure()
	plt.plot(x)
	plt.title('States')
	plt.figure()
	plt.plot(u)
	plt.title('Controls')
	plt.show()
	bp()
	DDP.gui.stopServing()



if __name__ == "__main__":
	main()

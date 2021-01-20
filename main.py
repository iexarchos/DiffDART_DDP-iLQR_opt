import numpy as np
from pdb import set_trace as bp
from DDP_opt import DDP_Traj_Optimizer
from envs.cart_pole import DartCartPoleEnv
from envs.snake_7link import DartSnake7LinkEnv
import diffdart as dart


def main():
	#Choose example and corresponding initial condition X0:

	#Env = DartCartPoleEnv
	#X0 = [0., 3.14, 0., 0.] 

	Env = DartSnake7LinkEnv
	X0 = None

	

	T = 2.0 # planning horizon in seconds

	maxIter = 20 # maximum number of iterations
	threshold = 0.001 # Optional, set to 'None' otherwise. Early stopping of optimization if cost doesn't improve more than this between iterations.

	DDP = DDP_Traj_Optimizer(Env=Env,T=T,X0=X0)
	x,u,cost = DDP.optimize(maxIter = maxIter, thresh=threshold)

	bp()
	c = DDP.simulate_traj(x, u, render = True)
	print(c)
	bp()
	DDP.gui.stateMachine().stopServing()



if __name__ == "__main__":
	main()

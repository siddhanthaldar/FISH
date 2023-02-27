from gym.envs.registration import register 

register(
	id='RobotPegInCup-v1',
	entry_point='gym_envs.envs:RobotPegInCupEnv',
	max_episode_steps=20,
	) 


register(
	id='RobotKeyInsertion-v1',
	entry_point='gym_envs.envs:RobotKeyInsertionEnv',
	max_episode_steps=15, #20,
	) 

register(
	id='RobotFlipBagel-v1',
	entry_point='gym_envs.envs:RobotFlipEnv',
	max_episode_steps=25,
	) 

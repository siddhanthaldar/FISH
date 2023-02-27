from gym.envs.registration import register 

register(
	id='DoorOpen-v1',
	entry_point='hello_envs.envs:DoorOpenEnv',
	max_episode_steps=25
)

register(
	id='DrawerOpen-v1',
	entry_point='hello_envs.envs:DrawerOpenEnv',
	max_episode_steps=23
)

register(
	id='LightSwitch-v1',
	entry_point='hello_envs.envs:LightSwitchEnv',
	max_episode_steps=18
)


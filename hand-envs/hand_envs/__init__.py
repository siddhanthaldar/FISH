from gym.envs.registration import register 


register(
	id='CubeFlip-v1',
	entry_point='hand_envs.envs:CubeFlipEnv',
	max_episode_steps=35,
)


register(
	id='CubeSlide-v1',
	entry_point='hand_envs.envs:CubeSlideEnv',
	max_episode_steps=18,
)

register(
	id='BottleSpin-v1',
	entry_point='hand_envs.envs:BottleSpinEnv',
	max_episode_steps=48
)

register(
	id='CardSlide-v1',
	entry_point='hand_envs.envs:CardSlideEnv',
	max_episode_steps=36
)
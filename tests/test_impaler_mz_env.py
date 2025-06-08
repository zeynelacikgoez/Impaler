from envs import ImpalerMZEnv
from pettingzoo.test import api_test
def test_env_basic_step():
    env = ImpalerMZEnv(num_producers=1, num_consumers=1, max_steps=3)
    env.reset()
    for agent in env.agent_iter(3):
        obs, reward, terminated, truncated, info = env.last()
        act = None if terminated or truncated else env.action_space(agent).sample()
        env.step(act)


def test_pettingzoo_compliance():
    env = ImpalerMZEnv(num_producers=1, num_consumers=1, max_steps=3)
    api_test(env, num_cycles=50)

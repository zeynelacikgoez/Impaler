from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from gymnasium import spaces
# `agent_selector` may be exported as a module in recent PettingZoo versions
# and the class needs to be imported explicitly.
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv
from pettingzoo.test import api_test


def env_state_to_dict(state: np.ndarray) -> Dict[str, float]:
    return {f"s{i}": float(v) for i, v in enumerate(state)}


class ImpalerMZEnv(AECEnv):
    """Minimal PettingZoo AECEnv for the Impaler project."""

    metadata = {"name": "impaler_mz_env"}

    def __init__(self, num_producers: int = 2, num_consumers: int = 2, max_steps: int = 24,
                 reward_weights: Dict[str, float] | None = None) -> None:
        super().__init__()
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.max_steps = max_steps
        self.possible_agents: List[str] = [f"producer_{i}" for i in range(num_producers)] + \
            [f"consumer_{i}" for i in range(num_consumers)] + ["grid"]
        self.agents: List[str] = []
        self._agent_selector = agent_selector(self.possible_agents)
        # state: [aggregated_supply, aggregated_demand, per_producer..., per_consumer...]
        self.state = np.zeros(2 + num_producers + num_consumers, dtype=np.float32)
        if reward_weights is None:
            reward_weights = {"supply": 1.0, "co2": 1.0, "fair": 1.0}
        self.reward_weights = reward_weights
        obs_shape = (self.state.size,)
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {}
        for agent in self.possible_agents:
            if agent.startswith("producer"):
                self.action_spaces[agent] = spaces.Discrete(5)
            elif agent.startswith("consumer"):
                self.action_spaces[agent] = spaces.Discrete(3)
            else:  # grid
                self.action_spaces[agent] = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.has_reset = False

    # PettingZoo API -----------------------------------------------------
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        self.agents = list(self.possible_agents)
        self._agent_selector = agent_selector(self.agents)
        self.state = np.zeros(self.state.size, dtype=np.float32)
        self.rewards = {a: 0.0 for a in self.agents}
        self.cumulative_rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._step_count = 0
        self.has_reset = True
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str):
        return self.state.copy()

    def step(self, action: int):
        if not self.has_reset:
            raise RuntimeError("Environment must be reset before stepping")
        agent = self.agent_selection
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            self._was_dead_step(action)
            return
        self.rewards = {a: 0.0 for a in self.agents}
        self._apply_action(agent, action)
        reward = self._calc_reward()
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] = 0.0
        next_agent = self._agent_selector.next()
        self._accumulate_rewards()
        if self._agent_selector.is_last():
            self._step_count += 1
            if self._step_count >= self.max_steps:
                for a in self.agents:
                    self.terminations[a] = True
        self.agent_selection = next_agent

    # ------------------------------------------------------------------
    def _apply_action(self, agent: str, action) -> None:
        """Apply agent action to the environment state."""
        if agent.startswith("producer"):
            prod_idx = int(agent.split("_")[1])
            val = int(action) - 2
            self.state[0] += float(val)
            self.state[2 + prod_idx] = float(val)
        elif agent.startswith("consumer"):
            cons_idx = int(agent.split("_")[1])
            val = int(action) - 1
            self.state[1] += float(val)
            self.state[2 + self.num_producers + cons_idx] = float(val)
        else:  # grid
            val = float(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else float(action)
            val = float(np.clip(val, -5.0, 5.0))
            self.state[0] += val

    def _calc_reward(self) -> float:
        deficit = abs(float(self.state[0] - self.state[1]))
        co2 = float(np.maximum(self.state[0], 0.0))
        fair = float(np.std(self.state[2:])) if self.state.size > 2 else 0.0
        r_supply = -deficit
        r_co2 = -co2
        r_fair = -fair
        total_reward = (
            self.reward_weights.get("supply", 1.0) * r_supply
            + self.reward_weights.get("co2", 1.0) * r_co2
            + self.reward_weights.get("fair", 1.0) * r_fair
        )
        return total_reward

    # Public helper ------------------------------------------------------
    def env_state(self) -> Dict[str, float]:
        """Return global state vector for CTDE setups."""
        return env_state_to_dict(self.state.copy())


if __name__ == "__main__":
    env = ImpalerMZEnv()
    api_test(env, num_cycles=5)

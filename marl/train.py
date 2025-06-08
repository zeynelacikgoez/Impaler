import argparse
import yaml
import os
from ray import tune
from ray.rllib.algorithms.qmix import QMIXConfig
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
import supersuit as ss

from envs import ImpalerMZEnv


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(config=None):
    if config is None:
        config = {}
    reward_cfg_path = config.get("reward_cfg", "marl/config/reward.yml")
    if os.path.exists(reward_cfg_path):
        reward_weights = load_yaml(reward_cfg_path)
    else:
        reward_weights = None
    return ImpalerMZEnv(reward_weights=reward_weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    register_env("impaler.MZEnv", lambda env_cfg: make_env(env_cfg))
    vec_env = ss.vectorize_aec_env(lambda: make_env(cfg), num_envs=cfg.get("num_envs", 1), num_cpus=min(cfg.get("num_envs", 1), 4))

    qmix_cfg = QMIXConfig().environment(lambda _: vec_env)
    qmix_cfg = qmix_cfg.training(mixer=cfg.get("model", {}).get("mixer", "qmix"))
    qmix_cfg = qmix_cfg.rollouts(num_rollout_workers=cfg.get("num_workers", 0))
    qmix_cfg = qmix_cfg.framework(cfg.get("framework", "torch"))
    qmix_cfg.update_from_dict(cfg)

    callbacks = []
    if args.wandb:
        callbacks.append(WandbLoggerCallback(project="impaler-marl", log_config=True))
    tune.run(cfg.get("alg", "QMIX"), config=qmix_cfg.to_dict(), callbacks=callbacks)


if __name__ == "__main__":
    main()

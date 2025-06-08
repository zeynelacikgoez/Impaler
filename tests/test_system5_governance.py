import pytest


def test_s5_updates_s4_policy_parameters(custom_model_factory):
    config_dict = {
        "vsm_on": True,
        "admm_on": True,
        "simulation_steps": 2,
        "goods": ["Food", "Energy"],
        "planning_priorities": {
            "fairness_weight": 0.8,
            "planwirtschaft_params": {
                "underproduction_penalty": 1.5,
                "co2_penalties": {"Energy": 7.5},
            },
        },
    }

    model = custom_model_factory(config_dict=config_dict, producers=1, regions=1)

    s4_planner = model.system4planner

    initial_lambda = s4_planner.lambdas.get("co2_Energy")
    assert initial_lambda is None or initial_lambda != 7.5

    model.stage_manager.run_stages(stages_to_run=["system5_policy"])

    assert s4_planner.lambdas["underproduction_Food"] == 1.5
    assert s4_planner.lambdas["underproduction_Energy"] == 1.5
    assert s4_planner.lambdas["co2_Energy"] == 7.5
    assert s4_planner.priority_config.fairness_weight == 0.8

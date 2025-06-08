def test_low_s3_effectiveness_adapts_s4_params(custom_model_factory, mocker):
    # 1. Setup: Create a model where S3 performance will likely be low
    model = custom_model_factory(regions=2, producers=4, vsm_on=True, admm_on=True)
    s3_manager = model.system3manager
    s4_planner = model.system4planner
    initial_rho = s4_planner.admm_config.rho # Store initial ADMM rho

    # (Optional) Mock RM performance to force low effectiveness in S3 report
    # This requires System3Manager to store the calculated effectiveness
    # Or mock the _update_coordination_effectiveness method
    mocker.patch.object(s3_manager, '_update_coordination_effectiveness', return_value=None) # Prevent update
    s3_manager.coordination_effectiveness = 0.4 # Force low effectiveness

    # 2. Run steps to trigger S3 -> S4 interface
    model.stage_manager.run_stages(stages_to_run=["system3_aggregation", "system2_coordination", "system3_feedback"]) # Run relevant S3 stages
    # Prepare report (assuming this happens in one of the stages or bookkeeping)
    report = s3_manager._prepare_operational_report()
    # Manually trigger S4 receiving the report (or run the full step/stage)
    s4_planner.receive_operational_report(report)

    # 3. Check S4 Reaction
    # S4's receive_operational_report should have adapted parameters
    assert s4_planner.admm_config.rho > initial_rho # Expect rho to increase

    # (Optional) Run another planning cycle and check if the new rho is used
    # model.step() # Run full next step
    # Check logs or internal state of S4/ADMM during the next planning run
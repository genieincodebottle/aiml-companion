RUN_EXPERIMENT = False   # True trains four more models on your T4; our runs are in the table of this step

if RUN_EXPERIMENT:
    experiment([("ipo-s0", "ipo", 5.0, 0, {"every": 125}),                  # TRL: DPOConfig(loss_type="ipo", beta=5.0)
                ("simpo-s0", "simpo", 2.0, 0, {"gamma": 1.0, "every": 125}),  # TRL: CPOConfig(loss_type="simpo", cpo_alpha=0), experimental
                ("orpo-s0", "orpo", 0.1, 0, {"every": 125}),                # TRL: ORPOConfig(beta=0.1), experimental
                ("kto-s0", "kto", 0.1, 0, {"every": 125})],                 # TRL: KTOConfig(beta=0.1), stable
               dpo_pairs, ref_dpo)

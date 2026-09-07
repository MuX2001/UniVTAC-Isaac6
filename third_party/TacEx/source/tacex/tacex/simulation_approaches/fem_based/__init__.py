try:
    import tacex_uipc

    from .mani_skill_sim import ManiSkillSimulator
    from .mani_skill_sim_cfg import ManiSkillSimulatorCfg

    __all__ = ["ManiSkillSimulator", "ManiSkillSimulatorCfg"]
except ImportError as exc:
    raise ImportError("FEM tactile simulation dependencies failed to import") from exc

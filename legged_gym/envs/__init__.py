from .Go2_Stand.Go2_Handstand.Go2_Handstand import Go2_Handstand_Robot
from .Go2_Stand.Go2_Handstand.Go2_Handstand_Config import Go2_Handstand_Cfg_Yu, Go2_Handstand_PPO_Yu
from legged_gym.utils.task_registry import task_registry

task_registry.register("go2_handstand", Go2_Handstand_Robot, Go2_Handstand_Cfg_Yu(), Go2_Handstand_PPO_Yu())

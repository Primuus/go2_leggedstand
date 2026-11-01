# 工程范围

- 训练环境、奖励及参数以 yusongmin1/My_unitree_go2_gym 为来源。用户已授权复用 /home/ry/project/quadruped 和 /home/ry/project/quadruped_train 的 Go2 资产并参考本地 MuJoCo 部署；来源记入 doc/local_asset_manifest.json。不要移植旧策略的观测或训练参数。
- 目标为 Go2 后腿支撑、前腿离地的直立行走，任务为 go2_handstand，当前训练后端已适配为 Isaac Lab ManagerBasedRLEnv；Isaac Gym 源码保留作参考溯源。
- 提取基线的环境、奖励、控制和 PPO 超参数保持上游原样。实际修正必须说明原因并更新 doc/upstream_manifest.json。
- 用户已批准将 descire_joint_angles 调整为基座高 0.52 m、前脚高 0.67 m、后脚落地的参考姿态，见 doc/参考关节与高度目标对齐.md。用户进一步批准直接从该直立参考姿态初始化：default_joint_angles 与参考角一致，基座初始高 0.52 m、俯仰 −90°，重置关节角与机身速度不再随机化。奖励公式与权重未变。
- 用户已要求先关闭全部 domain_rand 开关。固定材料/关节名义参数见训练配置，质量和质心采用资产原值；观测噪声与命令采样保留。不得未经要求重新开启域随机化。
- 用户已批准借鉴本机 ry_unitree_rl_lab 的运行框架。新工程运行时不得依赖旧工程；保留 45/86 维观测、12 动作、参考 PPO 和 23 项奖励。启动使用 ./ry_python.sh scripts/train_lab.py，见 doc/IsaacLab适配与小步验证.md。
- 用户最新要求：适配和小规模验证结束后停止，不启动正式 1500 轮训练。
- 不下载参考仓库的 logs、datasets 或部署场景作为训练依赖。
- 本地 MuJoCo XML 的机械参数以本工程训练 URDF 为基准：固定连接部分按平行轴定理合并，保留质心、完整惯性张量和全部碰撞几何；不得恢复旧 XML 的小腿质量。资产变更更新 doc/local_asset_manifest.json。
- 文档放在 doc/；训练模型与实验日志保留但不提交；仅本地 Git 提交，不推送 GitHub。

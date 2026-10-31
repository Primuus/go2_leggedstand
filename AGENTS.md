# 工程范围

- 训练环境、奖励及参数以 yusongmin1/My_unitree_go2_gym 为来源。用户已授权复用 /home/ry/project/quadruped 和 /home/ry/project/quadruped_train 的 Go2 资产并参考本地 MuJoCo 部署；来源记入 doc/local_asset_manifest.json。不要移植旧策略的观测或训练参数。
- 目标为 Go2 后腿支撑、前腿离地的直立行走，任务 ID 为 go2_handstand，使用 Isaac Gym。
- 提取基线的环境、奖励、控制和 PPO 超参数保持上游原样。实际修正必须说明原因并更新 doc/upstream_manifest.json。
- 用户已批准将 descire_joint_angles 调整为基座高 0.52 m、前脚高 0.67 m、后脚落地的参考姿态，见 doc/参考关节与高度目标对齐.md。用户进一步批准直接从该直立参考姿态初始化：default_joint_angles 与参考角一致，基座初始高 0.52 m、俯仰 −90°，重置关节角与机身速度不再随机化。奖励公式与权重未变。
- 不下载参考仓库的 logs、datasets 或部署场景作为训练依赖。
- 本地 MuJoCo XML 的机械参数以本工程训练 URDF 为基准：固定连接部分按平行轴定理合并，保留质心、完整惯性张量和全部碰撞几何；不得恢复旧 XML 的小腿质量。资产变更更新 doc/local_asset_manifest.json。
- 文档放在 doc/；训练模型与实验日志保留但不提交；仅本地 Git 提交，不推送 GitHub。

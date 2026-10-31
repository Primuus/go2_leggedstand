# 工程范围

- 训练环境、奖励及参数以 yusongmin1/My_unitree_go2_gym 为来源。用户已授权复用 /home/ry/project/quadruped 和 /home/ry/project/quadruped_train 的 Go2 资产并参考本地 MuJoCo 部署；来源记入 doc/local_asset_manifest.json。不要移植旧策略的观测或训练参数。
- 目标为 Go2 后腿支撑、前腿离地的直立行走，任务 ID 为 go2_handstand，使用 Isaac Gym。
- 提取基线的环境、奖励、控制和 PPO 超参数保持上游原样。实际修正必须说明原因并更新 doc/upstream_manifest.json。
- 不下载参考仓库的 logs、datasets 或部署场景作为训练依赖。
- 本地 MuJoCo XML 的机械参数以本工程训练 URDF 为基准：固定连接部分按平行轴定理合并，保留质心、完整惯性张量和全部碰撞几何；不得恢复旧 XML 的小腿质量。资产变更更新 doc/local_asset_manifest.json。
- 文档放在 doc/；训练模型与实验日志保留但不提交；仅本地 Git 提交，不推送 GitHub。

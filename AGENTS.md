# 工程范围

- 本工程按用户最新要求，仅以 yusongmin1/My_unitree_go2_gym 为来源；不得参考或移植本地其他机器人训练工程。
- 目标为 Go2 后腿支撑、前腿离地的直立行走，任务 ID 为 go2_handstand，使用 Isaac Gym。
- 提取基线的环境、奖励、控制和 PPO 超参数保持上游原样。实际修正必须说明原因并更新 doc/upstream_manifest.json。
- 不下载参考仓库的 logs、datasets 或部署场景作为训练依赖。
- 文档放在 doc/；训练模型与实验日志保留但不提交；仅本地 Git 提交，不推送 GitHub。

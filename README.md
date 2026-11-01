# Go2 后腿直立行走

训练任务提取自 [yusongmin1/My_unitree_go2_gym](https://github.com/yusongmin1/My_unitree_go2_gym) 的 `go2_handstand`，当前已适配服务器现有 Isaac Lab，保留参考仓库自带 PPO、45/86 维观测与 23 项奖励。

当前入口为 `./ry_python.sh scripts/train_lab.py`。适配、短训练验证和实际运行说明见 [Isaac Lab 适配记录](doc/IsaacLab适配与小步验证.md)。用户要求验证后停止，尚未启动正式训练。

安装、训练、回放与源码分析见 [doc/工程说明.md](doc/工程说明.md)。逐文件来源与改动标记见 [doc/upstream_manifest.json](doc/upstream_manifest.json)。

已接入复用本机 Go2 资产的独立 MuJoCo 部署，见 [本地 sim2sim](doc/本地MuJoCo部署.md)。

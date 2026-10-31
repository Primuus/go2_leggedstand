# 本地 MuJoCo sim2sim

本入口运行当前 `go2_handstand` 的 45 维输入、12 维输出 TorchScript 策略。资产复用用户指定的 `/home/ry/project/quadruped`，部署参考其按名称映射关节/执行器、每个物理步计算 PD 的实现。两个旧工程均未修改，也不作为运行时依赖。

## 复用与适配

- 从 `quadruped/src/controller/rl/resources/robots/go2/` 复制 `go2.xml`、`scene_flat.xml` 和 16 个 OBJ 网格，约 27.11 MiB，保存在本工程 `resources/robots/go2/mujoco/`。这些都是本机文件复制，没有通过服务器链路搬运网格。
- 三份 Go2 URDF（本工程、quadruped、quadruped_train）的 SHA-256 均为 `4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4`。无需更换当前训练 URDF/DAE。
- 旧部署为 6×45=270 维 HIM 历史输入，且命令位于前 3 维；当前策略为单帧 45 维，角速度、重力、命令依次占前 9 维。不能复用旧权重或直接运行旧观测构造器。
- 当前训练配置由 `deploy_mujoco/config_loader.py` 直接加载配置源码，隔离配置继承导入，不执行训练任务注册。MuJoCo 环境无需安装 Isaac Gym，也无需安装本工程或旧工程 Python 包。
- 关节与动作顺序为 FL、FR、RL、RR，各腿 hip/thigh/calf。复用 XML 的执行器排列是 FR、FL、RR、RL；部署按名称查找地址，不能直接把 12 维策略动作写入原始执行器数组。
- 训练参数直接来自配置：Kp=40、Kd=1，action scale=0.25，物理步长 0.005 秒、decimation=4，50 Hz 策略。机身初始高度 0.42 米，关节初值使用训练默认值，关闭部署侧随机化。
- 复用 XML 的滑动摩擦从 0.4 改为参考仓库 MuJoCo 的 0.6；小腿执行器上限从 45.43 改为训练 URDF 的 35.55 N·m，髋/大腿保持 23.7 N·m。训练奖励和控制参数未修改。
- 当前模型总质量 15.206408 kg，训练 URDF 总质量 15.019 kg。此质量差异也存在于参考仓库的 MuJoCo 模型，本次保留原模型惯性；没有声称两个仿真器机械参数完全等价。其余接触与被动关节参数沿用参考/复用模型的固定参数，未复现训练域随机化。
- 未套用旧普通四足的 `gravity_z` 跌倒停止条件，因为直立时该分量接近零。当前只在数值异常时终止；报告是运行诊断，不是完整直立/步态验收。

来源、原始哈希和修改后哈希见 [local_asset_manifest.json](local_asset_manifest.json)。

## 运行

本机已存在的 `RL` 环境可运行，已确认 MuJoCo 3.8.1、PyTorch 2.12.0+cpu。无需新建环境或安装额外输入库。在工程根目录运行：

```bash
# 无策略也可检查模型、PD、接触仿真；此模式不会自主直立
/home/ry/miniconda3/envs/RL/bin/python deploy_mujoco/sim2sim_go2_handstand.py \
  --check --headless --duration 2

# 使用本工程训练并导出的 TorchScript 策略，打开 MuJoCo 窗口
/home/ry/miniconda3/envs/RL/bin/python deploy_mujoco/sim2sim_go2_handstand.py \
  --policy logs/go2_handstand/exported/policies/policy_1.pt \
  --speed 0.2 --yaw-rate 0 --duration 60

# 无窗口运行并保存诊断
/home/ry/miniconda3/envs/RL/bin/python deploy_mujoco/sim2sim_go2_handstand.py \
  --policy logs/go2_handstand/exported/policies/policy_1.pt \
  --headless --speed 0.2 --duration 20 \
  --report logs/sim2sim/forward.json
```

`--policy` 与 `--check` 必须二选一。`--policy` 读取导出的 TorchScript Actor，不是训练 checkpoint 字典。训练端使用 `legged_gym/scripts/play.py` 导出，详见 [工程说明](工程说明.md)。当前没有训练好的直立策略，不能使用旧 270 维策略代替。

窗口获取焦点后：W/S 增减前向命令，Q/E 增减转向命令，空格清零命令，R 重置。前向键盘范围 −0.2～0.6 m/s，转向 −1～1 rad/s；横向为零。命令行可显式指定其他有限速度用于测试。键盘使用 MuJoCo 窗口回调，不需要 evdev 权限、pygame 或 pynput。

`--headless` 尽快执行固定仿真时长；有窗口时按实际时间同步，策略与 PD 频率相同。报告包含命令、世界坐标位移、平均机身高度和直立余弦；重置后统计重新开始。直立余弦取重力投影的负 X 分量，接近 1 表示目标方向。

## 已完成验证及限制

- 复用场景正常加载，2 秒无策略 PD 物理运行数值有限。
- 12 个关节顺序、名称映射、关节角限位和执行器扭矩限位与训练 URDF 一致。
- 在 5 组不同姿态、关节状态下，与参考环境实际 `compute_observations` 的 45 维结果一致，容差 2e−7；Critic 输入为 86 维。部署进程未导入 Isaac Gym。
- 临时导出原网络结构的随机初始化 Actor，验证 TorchScript 加载、推理和 0.2 秒物理循环；270 维旧接口被拒绝。临时测试权重已删除，没有用随机策略宣称训练效果。
- EGL 离屏渲染用于检查场景和网格；交互窗口/键盘仍需用户图形会话实际体验。
- 尚无训练好的直立策略，未验证直立行走成功率。服务器 SSH 仍返回认证失败，新增工程尚未同步到容器。

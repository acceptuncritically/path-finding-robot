# C 端（机器人躯体底层）· 放置目录

> 本目录预留给 **C 语言实现的机器人躯体端**（由队友负责），与仓库根目录的
> [`brain/`](../brain/)（Python 大脑端）平级，两者只通过固定 JSON 协议通信。

## 职责边界

| 模块 | 负责方 | 内容 |
|---|---|---|
| **Python 大脑端** | 本仓库 `brain/` | 自然语言输入、调用大模型、坐标解析与校验、下发目标坐标、接收并解析状态/路径、人话反馈 |
| **C 躯体端（本目录）** | 队友 | 串口/网络通信、A* 栅格寻路、视觉避障（摄像头/传感器）、电机控制、状态与路径回传 |

## 固定通信协议（字段冻结，详见 [`docs/Python大脑端-任务总结与C端通信协议说明.md`](../docs/Python大脑端-任务总结与C端通信协议说明.md)）

坐标系：**6×6 栅格，左上角 (0,0)，x 向下（行）、y 向右（列），坐标范围 0~5 的整数**。

| 方向 | 报文 |
|---|---|
| Python → C（下发目标） | `{"target_x": 5, "target_y": 5}` |
| C → Python（寻路成功） | `{"status":"running","path":[{"x":0,"y":0},{"x":1,"y":0}]}` |
| C → Python（寻路失败） | `{"status":"fail","path":[]}` |

传输约定：**一条报文一行，UTF-8，以 `\n` 结尾**；一次只处理一个任务（收到下发 → 回一条结果）。

## 建议的 C 工程结构（队友按需调整）

```text
c_controller/
├── README.md          # 本文件
├── src/
│   ├── main.c         # 入口：串口/网络收发主循环
│   ├── protocol.c/.h  # JSON 报文解析与组装（对应上面的固定协议）
│   ├── astar.c/.h     # 6×6 栅格 A* 寻路
│   ├── vision.c/.h    # 视觉避障（摄像头/传感器）
│   └── motor.c/.h     # 电机控制
├── include/
└── tests/             # 单元测试 / 联调脚本
```

## 联调前需要与 Python 端确认的问题

1. 串口/TCP 帧边界：每条 JSON 以 `\n` 结尾是否可行？串口参数（默认 115200, 8N1）？
2. `path` 是否包含起点和终点？`path[0]` 是当前位姿吗？成功时 `path` 是否可能为空？
3. 小车到达终点后是否还会回包？能否补一条 `{"status":"arrived","path":[]}` 作为任务结束信号？
4. A* 规划最长耗时？Python 端等待回包超时默认 5 秒是否足够？
5. 除 `running` / `fail` 外是否还有别的状态或错误帧？
6. C 端解析 JSON 对空格、字段顺序、换行是否敏感？（Python 端实际下发紧凑单行报文，首条联调时人工核对字节）

联调工具：先在仓库根目录运行模拟 C 端服务器做彩排：

```powershell
cd E:\test1\robot_search_path
python tools\mock_c_server.py --port 9000
python -m brain --transport tcp --tcp-port 9000
```

真实联调时把 `--transport` 换成 `serial`（优先方案，需先 `pip install pyserial`）。
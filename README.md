# 具身 Agent 小车 · 自然语言视觉避障机器人
## Python 大脑端（LLM Brain）· 仓库 robot_search_path

> 基于大语言模型的自然语言指令式视觉避障机器人 —— Python 大脑端：
> **自然语言 → 大模型 → 目标栅格坐标 → 固定 JSON 发给 C 端 → 接收状态/路径 → 反馈成人话。**
>
> 仓库分层：`brain/`（Python 大脑端，本仓库已实现）+ `c_controller/`（C 端躯体端，队友实现，见占位 README）+ `docs/`（协议与任务文档）。

本仓库的 Python 大脑端是可直接运行的**交付级实现**（不是伪代码），对应开发路线图的全部 4 步：

| 路线图 | 状态 | 对应实现 / 验证方式 |
|---|---|---|
| ① 基础 demo：一句话 → 大模型 → 坐标 → 打印 JSON | ✅ 已实现 | `python -m brain`（离线 MockLLM，无需 API Key） |
| ② 回包解析：样例 JSON → 解析 status/path → 人话反馈 | ✅ 已实现 | `python -m brain --selftest`（9 个场景全自动验证） |
| ③ 联调：真实串口（优先）/ TCP（仿真）收发 | ✅ 已实现 | `--transport tcp` / `--transport serial` + `tools/mock_c_server.py` 彩排 |
| ④ 优化：提示词、异常处理、交互体验 | ✅ 已实现 | 重试/校验/超时/日志/连续对话/148 个单元测试 |

---

## 1. 职责边界

| 模块 | 负责方 | 职责 |
|---|---|---|
| **LLM 大脑端（本目录，Python）** | 你 | 自然语言输入、调用大模型、语义解析、坐标校验、组装/下发 JSON、接收/解析 C 端回报、人话反馈、连续对话 |
| **机器人躯体端（C 语言）** | 队友 | 串口/网络通信、A* 栅格寻路、视觉避障、电机控制、回传状态与路径 |
| **双方接口** | 共同 | 只认两条固定 JSON 报文（字段冻结，见第 3 节） |

大脑端**不碰** A*、视觉、电机；C 端**不碰**大模型。两侧只通过 JSON 解耦，可并行开发、独立替换。

---

## 2. 目录结构（按“子系统 → 代码分层 → 测试”分级）

```text
robot_search_path/                       # PyCharm 工程根目录（本仓库）
├── README.md                            # 本文件：总览 + 快速开始 + 联调
├── .gitignore                           # 忽略 .env / 日志 / 缓存 / .venv
├── .env.example                         # 配置模板（大模型 Key / 通信方式 / 日志）
├── requirements.txt                     # 核心零依赖；串口联调时可选装 pyserial
│
├── docs/                                # 文档层
│   └── Python大脑端-任务总结与C端通信协议说明.md
│                                        # 任务清单 + 协议规范（与 C 端联调依据）
│
├── brain/                               # ★ Python 大脑端源码（按层分包）
│   ├── __init__.py / __main__.py        # python -m brain 入口
│   ├── config.py                        # 配置层：环境变量 + .env（业务代码不读 os.environ）
│   ├── errors.py                        # 异常体系：配置/大模型/协议/传输/会话
│   ├── protocol.py                      # 协议层：与 C 端报文组装/解析/严格校验（无 IO）
│   ├── grid.py                          # 栅格工具：6×6 BFS（给模拟 C 端用）
│   ├── prompt.py                        # 提示词层：系统提示词 + 重试纠正提示
│   ├── planner.py                       # 规划层：自然语言 → 目标坐标（调用/解析/重试/校验）
│   ├── feedback.py                      # 表现层：人话反馈模板（可替换/多语言）
│   ├── session.py                       # 会话层：任务状态机（与 UI 无关，可被 Web/语音复用）
│   ├── logging_setup.py                 # 基础设施：控制台 + 文件日志
│   ├── cli.py                           # 入口层：交互 / --once / --selftest
│   ├── llm/                             # 大模型适配层
│   │   ├── base.py                      # LLMClient 抽象
│   │   ├── mock.py                      # 离线规则 MockLLM（可注入脚本化输出）
│   │   ├── openai_compat.py             # 通义千问 / LLaMA / Ollama 兼容客户端（零第三方依赖）
│   │   └── parser.py                    # 大模型输出容错解析 + 严格校验
│   └── transport/                       # 传输适配层
│       ├── base.py                      # Transport 抽象（send_line / recv_line / drain）
│       ├── mock.py                      # 本地模拟 C 端（BFS + 异常场景注入）
│       ├── tcp.py                       # TCP 客户端（按行分帧，处理粘包/半包/空行）
│       └── serial.py                    # 串口（延迟导入 pyserial，未安装时给安装提示）
│
├── c_controller/                        # C 端（机器人躯体底层）放置目录 —— 待队友实现
│   └── README.md                        # C 端职责与协议接口说明（占位）
│
├── tools/
│   └── mock_c_server.py                 # 模拟 C 端的 TCP 服务器（无硬件联调彩排）
│
└── tests/                               # 测试层：148 个单元/集成测试（标准库 unittest）
    ├── test_protocol.py                 # 报文格式/字段校验（含 bool/浮点/越界等陷阱）
    ├── test_llm_parser.py               # 大模型输出解析（围栏/解释/歧义/兜底）
    ├── test_planner.py                  # 重试策略与失败归类
    ├── test_feedback.py                 # 人话反馈文案
    ├── test_config.py                   # 配置加载 / BOM / 优先级 / 非法值
    ├── test_session.py                  # 状态机：成功/失败/超时/异常/并发/迟到回包
    ├── test_transport_mock.py           # Mock C 端行为
    ├── test_transport_tcp.py            # TCP 真 socket 集成测试
    └── test_cli.py                      # CLI / selftest / --once / 退出码
```

### 2.1 代码分层与依赖方向（解耦原则）

| 层 | 位置 | 职责 | 依赖方向 |
|---|---|---|---|
| 入口 / 表现层 | `brain/cli.py`、`brain/feedback.py` | 命令行交互、中文反馈文案 | → 会话层 |
| 会话层 | `brain/session.py` | 任务状态机、串行保护、异常归类、残留回包清理 | → 规划层 / 传输抽象 / 反馈 |
| 规划层 | `brain/planner.py`、`brain/llm/` | 自然语言 → 目标坐标、重试、解析安全边界 | → 协议层 / 提示词 / LLM 抽象 |
| 协议层 | `brain/protocol.py` | 与 C 端的固定 JSON 协议（组装/解析/校验，纯函数无 IO） | → 异常体系 |
| 传输层 | `brain/transport/` | Mock / TCP / 串口收发，按行分帧 | → 协议常量 / 异常体系 |
| 基础设施 | `brain/config.py`、`logging_setup.py`、`errors.py` | 配置、日志、异常体系 | 无业务依赖 |

依赖方向始终是**入口 → 会话 → 规划 / 传输 → 协议 / 基础设施**，不反向依赖；
`LLMClient` 与 `Transport` 都是抽象接口，Mock / 真实实现可随时替换，业务代码不改（详见第 9 节）。

---

## 3. 通信协议（与 C 端冻结，禁止改字段）

**坐标系**：6×6 栅格；**左上角 (0,0)**；**x 向下（行号）**、**y 向右（列号）**；坐标范围 **0~5 的整数**。

| 方向 | 报文 |
|---|---|
| Python → C（下发） | `{"target_x": 5, "target_y": 5}` |
| C → Python（成功） | `{"status":"running","path":[{"x":0,"y":0},{"x":1,"y":0}]}` |
| C → Python（失败） | `{"status":"fail","path":[]}` |

传输约定：**一条报文一行，UTF-8，以 `\n` 结尾**；一次只处理一个任务（发一条、等回包、再发下一条）。
实际下发的字节是**紧凑单行**：`{"target_x":5,"target_y":5}\n`（上表带空格仅为可读性；JSON 语义一致）。
详细规范与待确认边界问题见 [通信协议与任务说明](<docs/Python大脑端-任务总结与C端通信协议说明.md>) 第 3、4、6 章。

---

## 4. 快速开始（零依赖，离线可跑）

要求：**Python 3.10+**（开发验证用 3.12；本 PyCharm 工程自带的 `.venv` 为 Python 3.13，也可以直接用）。核心功能**只用标准库**，不需要 `pip install` 任何东西。

```powershell
cd E:\test1\robot_search_path
$env:PYTHONUTF8='1'      # Windows 控制台建议开启，避免中文乱码

# ① 交互模式（MockLLM + Mock C 端，无需 API Key / 硬件）
python -m brain

# ② 内置自检：成功/失败/超时/坏 JSON/未知状态/越界重试等 9 个场景
python -m brain --selftest

# ③ 单次执行（脚本化 / CI 用）
python -m brain --once "去迷宫右下角"

# ④ 单元测试（148 个：单元 + TCP 真 socket 集成）
python -m unittest discover -s tests -v
```

`--once` 退出码约定：`0` = 任务完成（含 C 端返回 `fail`）；`1` = 任务未完成（无法解析 / 通信异常 / 超时 / 协议错等）；`2` = 配置或连接失败。适合放进 CI 或脚本判断。

**交互模式示例**（`python -m brain`）：

```text
请输入指令 > 去迷宫右下角

[大脑] 目标解析完成，目标点 (x=5, y=5)（第 6 行第 6 列）；已规划路线，小车准备出发，
      路径：[(0,0),(1,0),...,(5,5)]（共 10 步）

请输入指令 > 去(1,1)          ← (1,1) 是障碍物

[大脑] 目标点被障碍物阻挡，无法规划可行路线，请更换目标。

请输入指令 > q
已退出，再见。
```

---

## 5. 接真实大模型（第 4 步优化：提示词 + 容错）

任何 **OpenAI 兼容接口**都能直接用：通义千问（DashScope 兼容模式）、Ollama、llama.cpp server、vLLM……

### 5.1 通义千问（云端）

```powershell
$env:BRAIN_LLM_PROVIDER='openai'
$env:BRAIN_LLM_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:BRAIN_LLM_API_KEY='sk-你的Key'       # 也可用 DASHSCOPE_API_KEY
$env:BRAIN_LLM_MODEL='qwen-plus'
python -m brain
```

### 5.2 本地 LLaMA（如 Ollama）

```powershell
$env:BRAIN_LLM_PROVIDER='openai'
$env:BRAIN_LLM_BASE_URL='http://127.0.0.1:11434/v1'
$env:BRAIN_LLM_MODEL='qwen2.5:7b'
$env:BRAIN_LLM_API_KEY='ollama'          # 本地服务可填任意非空值
python -m brain
```

### 5.3 也可以只用命令行参数 / `.env`

```powershell
python -m brain --llm openai --base-url "http://127.0.0.1:11434/v1" --model "qwen2.5:7b"
copy .env.example .env    # 然后编辑 .env（.env 已在 .gitignore 中，不会被提交）
```

**容错能力**（已内置并有单测覆盖）：

- 模型输出带 ```` ```json ```` 围栏、前后带解释文字 → 自动提取；
- 坐标是字符串 `"5"` / 浮点 `5.0` / 布尔 `true` / `null` → 拒绝并重试；
- 坐标越界（如 6、-1）→ 拒绝、带着错误原因重试最多 2 次；仍失败则提示用户，**绝不把非法数据发给 C 端**；
- 模型返回**整数** `-1,-1`（约定“无法确定”）→ 不重试，直接提示用户补充坐标/方位；浮点 `-1.0` 不算该标记，会按类型错误重试；
- **安全边界**：输出里出现多个不同目标坐标 → 视为歧义，重试而不是猜一个；文本明确说“无法确定”且没有合法目标 JSON → 不猜坐标；正则兜底只对“整段近似纯字段表达式”生效，避免把示例/解释里的坐标当目标下发；
- 网络超时 / HTTP 4xx/5xx → 客户端层重试（指数退避），最终失败给中文提示；
- 所有原始输出都进日志，便于调提示词。

---

## 6. 联调（第 3 步）：先 TCP 彩排，再上串口

### 6.1 TCP 彩排（不需要硬件，先验证协议收发）

准备两个终端：

```powershell
# 终端 A：启动“模拟 C 端”服务器（默认监听 127.0.0.1:9000）
cd E:\test1\robot_search_path
python tools\mock_c_server.py --port 9000

# 终端 B：大脑端连过去（交互模式）
cd E:\test1\robot_search_path
$env:PYTHONUTF8='1'
python -m brain --transport tcp --tcp-port 9000

# 或者只跑一条指令验证链路：
python -m brain --transport tcp --tcp-port 9000 --once "去迷宫右下角"
```

模拟 C 端支持异常场景，用来验证大脑端鲁棒性：

```powershell
python tools\mock_c_server.py --mode fail           # 总是返回 fail
python tools\mock_c_server.py --mode timeout        # 故意不回包（验证超时）
python tools\mock_c_server.py --mode bad_json       # 返回非法 JSON
python tools\mock_c_server.py --mode unknown_status # 返回未知 status
python tools\mock_c_server.py --mode noise          # 先发一个空行再回包（验证帧噪声处理）
python tools\mock_c_server.py --obstacles "1,1;1,2;2,4" --start "0,0"
```

### 6.2 串口（优先方案，最终上车用）

```powershell
pip install pyserial
python -m brain --transport serial --serial-port COM3 --baudrate 115200
# 或在 .env 里配置 BRAIN_TRANSPORT=serial / BRAIN_SERIAL_PORT=COM3
```

串口参数默认 `115200, 8N1`，每条报文 `\n` 结尾。若没装 `pyserial`，会给出明确安装提示而不会崩溃。

### 6.3 与 C 端联调前确认清单

`path` 是否含起点/终点、是否有“到达终点”结束信号、帧边界、超时阈值、JSON 空白敏感度等
—— 见 [Python大脑端-任务总结与C端通信协议说明.md](<docs/Python大脑端-任务总结与C端通信协议说明.md>) 第 6 章（可直接复制发给 C 端队友）。

---

## 7. 异常处理矩阵

| 场景 | 系统行为 | 用户看到的中文反馈 |
|---|---|---|
| 模型输出格式错/类型错/越界 | 带错误原因自动重试 ≤ 2 次，仍失败则不下发 | “无法理解这条指令…” / “坐标超出迷宫范围…” |
| 模型无法确定目标（-1,-1） | 不重试 | “无法确定目标位置，请给出具体坐标或方位。” |
| 大模型网络/鉴权失败 | 客户端指数退避重试，最终失败 | “大模型调用失败：… 请检查网络或 API 配置。” |
| C 端返回 `running` | 解析路径、格式化步数 | “目标解析完成…路径：[(0,0),…]（共 N 步）” |
| C 端返回 `fail` | 提示换目标 | “目标点被障碍物阻挡，无法规划可行路线，请更换目标。” |
| 等待回包超时（默认 5s，可配） | 回到 IDLE，可继续输入 | “小车未在 5 秒内响应，请检查串口/网络连接后重试。” |
| 超时后 C 端**迟到回包** | 新任务下发前自动排空残留报文，避免被误当成新任务结果（防“目标 A 却展示 B 的路径”） | 日志记“已丢弃 N 条残留/迟到报文”，用户正常看到新任务反馈 |
| 空行/帧噪声（C 端多发一个 `\n`） | TCP/串口自动跳过空行，继续等真正报文 | 无感 |
| C 端回包非法 JSON / 缺字段 | 记录原始报文，不崩溃 | “小车回传数据格式异常（原因：…），已记录原始报文。” |
| C 端回未知 status | 记录日志，向前兼容不崩溃 | “小车回传了未知状态：… 请检查小车程序版本。” |
| TCP/串口断开 | 标记连接失效并提示通信异常（当前版本需重启程序/重建会话，自动重连为后续项） | “串口/网络通信异常：… 请检查连接后重试。” |
| 并发提交（Web 场景） | 状态机串行保护，拒绝并发 | “上一次任务还没有结束，请等收到反馈后再输入新指令。” |

---

## 8. 测试

```powershell
python -m unittest discover -s tests -v
# 结果：Ran 148 tests ... OK
```

测试覆盖：

- **协议层**：下发报文逐字符对照、`bool/字符串/浮点/越界/缺字段/未知 status/空路径/深嵌套 JSON` 等边界；
- **大模型解析**：代码围栏、解释文字、纯字段兜底、`-1,-1` 约定、`-1.0` 不误判、多个不同目标视为歧义、示例坐标不误下发、类型/范围拒绝；
- **规划器**：首次成功、格式错重试、越界重试、重试用尽、LLM 故障、空指令；
- **会话状态机**：成功/失败/超时/坏 JSON/未知状态/通信异常/连续对话/会话关闭/并发保护/**迟到回包防错配**；
- **传输层**：Mock BFS 路径连通性、不可达、超时、脚本化回包；TCP 真 socket 收发、连接拒绝、对端断开（并标记失效）、空行噪声跳过；
- **配置**：`.env` 解析、UTF-8 BOM 兼容、环境变量优先级、非法值报错、不可变配置覆盖；
- **CLI**：`--selftest`、`--once`、`--version`、组件组装、错误退出码。

---

## 9. 解耦设计（为什么“能换、能测、能交付”）

```
                ┌──────────────────────────────────────────────┐
                │ CLI / 未来的 Web API / 语音入口              │
                └───────────────────┬──────────────────────────┘
                                    │ 只调 session.handle_instruction()
                ┌───────────────────▼──────────────────────────┐
                │ BrainSession（状态机 + 串行保护）            │
                │   依赖注入：planner / transport / feedback   │
                └──────┬──────────────────────────────┬────────┘
                       │                              │
        ┌──────────────▼─────────────┐   ┌────────────▼────────────────┐
        │ TargetPlanner              │   │ Transport（抽象）           │
        │   └─ LLMClient（抽象）     │   │   ├─ MockRobotTransport     │
        │        ├─ MockLLM          │   │   ├─ TcpTransport           │
        │        └─ OpenAICompatLLM  │   │   └─ SerialTransport        │
        └──────────────┬─────────────┘   └─────────────────────────────┘
                       │
        ┌──────────────▼─────────────┐   ┌─────────────────────────────┐
        │ protocol.py（值对象+校验） │   │ feedback.py（人话模板）     │
        └────────────────────────────┘   └─────────────────────────────┘
```

| 想替换什么 | 怎么做 | 需要改业务代码吗 |
|---|---|---|
| 换大模型（通义/LLaMA/本地） | 新增/配置 `LLMClient` 实现 | 不用 |
| 换通信方式（TCP/串口/蓝牙） | 新增 `Transport` 实现 | 不用 |
| 换交互方式（Web/语音） | 复用 `BrainSession`，自己负责输入输出 | 不用 |
| 改提示词 | 只改 `prompt.py` | 不用 |
| 改文案/做多语言 | 子类化 `FeedbackRenderer` | 不用 |
| 改协议字段 | 只改 `protocol.py`（需与 C 端同步，当前冻结） | —— |

另外：配置全部集中在 `config.py`（业务代码不读 `os.environ`）、异常按类型分层（便于 UI 决定重试/提示）、关键路径都有日志（联调排错）。

---

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| 控制台中文乱码 | 设置 `$env:PYTHONUTF8='1'`，或使用 Windows Terminal / VS Code 终端 |
| `串口功能需要 pyserial` | `pip install pyserial`（核心功能不需要它） |
| 串口打开失败 / 被占用 | 关闭串口助手等占用程序；确认 `--serial-port` 是设备管理器里的 COM 号 |
| TCP 连接失败 | 先确认 `tools/mock_c_server.py` 或 C 端程序已监听对应端口，`--tcp-host/--tcp-port` 一致 |
| 大模型 401/403 | 检查 `BRAIN_LLM_API_KEY` 是否正确、是否有模型权限 |
| 大模型 404 | 检查 `BRAIN_LLM_BASE_URL` 是否包含正确路径（如通义需 `/compatible-mode/v1`） |
| 模型总是输出废话 | 调大 `BRAIN_LLM_MAX_RETRIES`，优化 `prompt.py`，或换更强模型；用日志查看原始输出 |
| 回包解析总报错 | 打印 C 端原始报文；核对是否 `\n` 分帧、是否多了空格/中文、字段名是否为 `status/path/x/y` |
| 想调等待回包时间 | `--report-timeout 8` 或环境变量 `BRAIN_REPORT_TIMEOUT=8` |
| 想看详细链路日志 | `-v` 或 `BRAIN_VERBOSE=1`；`--log-file logs/brain.log` 落盘 |

---

## 11. 与规范文档的关系

- 本文档 = **代码使用说明**（怎么跑、怎么换、怎么联调）；
- [Python大脑端-任务总结与C端通信协议说明.md](<docs/Python大脑端-任务总结与C端通信协议说明.md>) = **任务与协议规范**（要做什么、报文定义、验收清单、大创申报素材）。

两者协议字段完全一致；若将来联调需要改字段，必须**同时改这两份文档 + `brain/protocol.py`**，并通知 C 端。

---

## 12. 已知边界与后续项

本实现已通过离线自检、148 个单元/集成测试、TCP 真 socket 彩排，并经独立审查（2 个联调前问题已修复并复测）：

1. **真实硬件未验证**：串口链路依赖 `pyserial` + 实际 C 端程序/小车硬件，需在联调阶段实测（当前环境无硬件）；
2. **真实大模型未联网验证**：OpenAI 兼容客户端已按 DashScope / Ollama 接口实现，但本环境未配 API Key 实测；首次联调请先用 `-v` 查看原始输出；
3. **断线不自动重连**：当前检测到 TCP/串口断开后标记连接失效并给中文提示，重新收发需重启程序/重建会话；后续可加自动重连（`Transport.reopen()` + 会话重试策略）；
4. **`path` 语义待 C 端确认**：是否含起点/终点、是否有“到达终点”结束报文等（见规范文档第 6 章），确认后可加“路径终点 == 目标点”的一致性校验作为第二道防线；
5. **相对指令**（“往前走两格”）暂不支持：协议只传绝对坐标，需要多轮上下文时可扩展；
6. **异常检测的顺序**：同一任务只处理一条回包，多余的报文会在下一任务前被 drain 并记日志；若 C 端会主动推送多条状态，需要双方约定 `seq`/`task_id` 后再扩展。
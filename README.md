# mini-muse / comic-strip

一个把想法变成长篇条漫的 agent：文本模型规划分页，逐页写分镜，图片模型逐格作画，
代码拼版渲染台词。一句话、零散想法、几段文字、整章小说、同一会话分次投喂——都收。

## 流程

```
你的输入
   │ ① 规划（POST /api/plan，只花几分钱文本费）
   │    输入分析 → kind（单页/连载）+ 页数 + 角色表 + 每页拍点 + 画师 + 预估花费
   ▼
确认卡（用户看页数/角色/成本/拍点，决定是否生成）
   │ ② 生成（GET /api/generate?sid=，SSE 逐页推进）
   │    单页 → 原 agent 流水线；连载 → 每页：写分镜 → 版式 → 补缺格 → 合成
   ▼
pages/*.png            一页一个分镜 slug（<base>-pNN），共用同一份剧本
```

一致性三保险（跨页/超长会话不靠把历史塞进上下文）：
风格卡与角色表在规划期定死、全部页共用一份剧本（生图自动拼卡）；
剧情靠"全部拍点 outline + 前情拍点摘要"逐页接力；上传参考图的角色按名字垫图。
同一会话再发一段文字，规划器看前情判断"续写（页码接着排）"还是"另起新作"。
单格单价用环境变量 `MINIMUSE_PRICE_PANEL` 配置（默认 ¥0.3），确认卡按它估价。

## 用法

Web 界面（推荐）：

```
.\run-web.bat        # 自动带密钥，起服务并打开浏览器 http://127.0.0.1:8765
                     # （.ps1 若被执行策略拦截，用 .bat 包装；或见下方一次性放开策略）
```

命令行两套本地启动脚本（已 git-ignore，内含密钥）：

```
.\run-deepseek.ps1 "你的想法"   # 主力：DeepSeek 分镜 + OpenAI 兼容中转的作画模型
.\run-glm.ps1     "你的想法"   # 免费档：glm-4.5-air 分镜 + cogview-3-flash 作画
.\run-ark.bat    "你的想法"   # 新主力：DeepSeek 编剧 + 字节 Seedream 作画（水印已关）
```

或手动配两组环境变量（文本/图片常常不是同一家）：

```
$env:OPENAI_API_KEY = "sk-..."          # 文本模型：分镜师
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
$env:MINIMUSE_MODEL = "deepseek-chat"
$env:IMAGE_API_KEY = "sk-..."            # 图片模型：画师
$env:IMAGE_BASE_URL = "https://你的OpenAI兼容中转/v1"   # 任何 OpenAI 兼容端点
$env:IMAGE_MODEL = "gpt-image-2"
.\.venv\Scripts\python comic.py "一个关于'追赶昨天的自己'的哲理短篇"
```

输出在 `comic_out\`：`scripts\*.json`、`storyboards\*.json`、`layouts\*.json`、`panels\*.png`、`pages\*.png`。

## 风格 Skill（塑造第一步的风格卡）

`skills/` 目录下每个 `.md` 是一个风格 skill：正文整体注入系统提示词，直接决定 save_script
产出的风格卡质量。内置：`ink-wash`（水墨淡彩）、`bw-cartoon`（2D黑白卡通）、`photoreal`
（写实摄影）、`us-comic`（美式彩漫）、`manga`（日式漫画）。

```
.\.venv\Scripts\python comic.py --list-styles              # 列出可选
.\.venv\Scripts\python comic.py "想法" --style photoreal   # 指定风格
```

Web 界面：输入框旁下拉选择风格；点「＋风格」上传自己的 `.md`（首个 `# 标题` 为显示名，
正文写核心词/措辞要求/禁止项，保存到 skills/ 即时生效）。

生成前先规划：点「生成/规划续写」会先出一张确认卡——页数、角色表、画师、
预估花费、分页拍点，确认才开始烧图；取消不花钱。在会话视图里输入则进入续写模式，
规划器自动判断是否衔接本会话的作品。

## 设计决策（踩过的坑）

1. **文字绝不进图**：图模型画字必乱，台词由 PIL 在拼版时代码渲染——每格提示词自动追加 "no text" 后缀
2. **风格卡全局复用**：分镜里一条全局 style，每格提示词自动拼上它——画风一致性的第一道保险（更彻底要靠角色表/参考图）
3. **格子规格三档**：`wide/square/tall` 写在分镜里（叙事规格），像素档位在工具层映射——出图比例=成图比例，拼版只缩不裁；换画师只换映射表
4. **错误信息是给模型的提示词**：环境类错误要写明"改参数救不了"，否则模型会傻傻重试（真实踩坑：httpx 缺失时模型改了 8 次提示词）
5. **国内 API 直连**：系统代理 + truststore 在 Windows 上做 TLS 握手会原生崩溃，run 脚本里已清掉代理变量

## 结构

| 文件 | 职责 |
|---|---|
| minimuse/schema.py | 线协议：消息/工具调用/结果；坏 JSON 落 `__raw__` 兜底 |
| minimuse/llm.py | 文本模型抽象：`BaseLLM` / `OpenAILLM`（任意 OpenAI 兼容端点） |
| minimuse/imagegen.py | 图片模型抽象：`BaseImageGen` / `OpenAIImageGen` + 字体/折行 |
| minimuse/agent.py | 主循环：max_steps、空回复纠正、步数耗尽强制收尾 |
| minimuse/tools.py | 工具骨架：`BaseTool` / `safe_execute` / `ToolCollection` |
| minimuse/comic_tools.py | 漫画三工具 + 规格映射表 + 分镜校验/拼版 + 多页 slug |
| minimuse/plan.py | 规划层：输入分析/页数/角色表/拍点/画师/成本预估 |
| minimuse/adapt.py | 长文本逐页流水线：写分镜→版式→补缺格→合成，页幂等可续 |
| comic.py | 命令行入口 |
| web.py | Web 服务：SSE 实时进度 + 历史作品 + 静态文件 |
| web/index.html | 单页界面（Vue 3 本地 vendor，零构建零外部依赖） |
| tests/test_comic.py | 工具链测试（测试内的纯色图替身 + 真分镜/拼版逻辑） |

## 测试

```
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff.exe check minimuse tests comic.py
```

包里没有任何假模型；测试需要的替身只存在于测试文件内部。

## 对照 nanoMuse 阅读（学习来源）

| mini-muse | nanoMuse |
|---|---|
| minimuse/schema.py | nanomuse/schema.py |
| minimuse/llm.py | nanomuse/llm/base.py + openai_chat.py |
| minimuse/tools.py | nanomuse/tools/base.py |
| minimuse/agent.py | nanomuse/agent/core.py::run() |
| （阶段 2 待写）sentinel.py | nanomuse/sentinel/policy.py、gate.py::guard() |

## 路线（下一步）

1. **Sentinel**：`generate_panel` 是花钱工具——按次计数、超预算审批、JSONL 审计每一笔提示词
2. **逐格人审**：每格生成后终端确认（保留/重画/跳过），烧钱前的最后一道闸
3. **角色表**：分镜加 `characters` 字段，出图自动拼角色卡——一致性的第二道保险
4. **网格页版式**：2×3 页模板，传统漫画页；仍是纯代码排版

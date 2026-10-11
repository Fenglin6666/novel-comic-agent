r"""novel-comic-agent：把一个想法变成一页原创漫画（版式分格 + 气泡台词）。

最简单的用法（本地脚本自动带密钥，已 git-ignore）：
  .\run-deepseek.bat "你的想法"   主力：DeepSeek 编剧分镜 + OpenAI 兼容中转的作画模型
  .\run-glm.bat     "你的想法"   免费档：glm-4.5-air + cogview-3-flash
  .\run-web.bat                  Web 界面（浏览器自动打开）

手动配置（文本/图片常常不是同一家）：
  文本模型  OPENAI_API_KEY / OPENAI_BASE_URL / MINIMUSE_MODEL
  图片模型  IMAGE_API_KEY / IMAGE_BASE_URL / IMAGE_MODEL（缺省回落 OPENAI_*）
  .\.venv\Scripts\python comic.py "一个关于'追赶昨天的自己'的哲理短篇"

输出在 comic_out\：scripts\*.json 剧本、storyboards\*.json 分镜、
layouts\*.json 版式、panels\*.png 逐格图、pages\*.png 成品页。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from novel_comic_agent.agent import Agent
from novel_comic_agent.comic_tools import (
    ComposePage,
    GeneratePanel,
    PlanLayout,
    SaveScript,
    SaveStoryboard,
    size_map_for,
)
from novel_comic_agent.imagegen import OpenAIImageGen
from novel_comic_agent.llm import OpenAILLM
from novel_comic_agent.skills import apply_skill, choose_skill, list_skills
from novel_comic_agent.refs import refs_prompt
from novel_comic_agent.tools import ToolCollection

COMIC_PROMPT = """你是 novel-comic-agent，一个把想法变成一页原创漫画的 agent
（版式分格，气泡台词由代码渲染）。

工作流，严格按此顺序：
1. save_script —— 剧本：标题；一条全局风格卡 style（所有格共享）；
   characters（1-3 个角色，每个 name + look 外貌卡：年龄/体型/发型/服装/标志性特征
   ——这是每一格人物一致性的依据）；synopsis（2-3 句梗概）。风格卡和外貌卡都用中文写。
2. save_storyboard —— 一页漫画的分镜。从版式模板里选 layout：
   action4（不规则斜切，动作/张力页）、big_open（大开页 + 三连格）、classic（规整 2×3）、
   trio/duo/solo/five/grid8；给 "auto" 则按格数自动选。
   格数必须等于模板槽数（solo 1、duo 2、trio 3、action4/big_open 4、five 5、classic 6、grid8 8）。
   每格字段：n、scene（这格发生了什么）、caption（简短旁白，可空）、
   dialogue（[{speaker, text, anchor: tl/tr/bl/br/c}]）、cast（出场角色，缺省=说话者）、
   prompt（中文，只描述画面里可见的东西：主体/构图/光线；抽象想法必须翻译成具体的视觉隐喻）。
3. plan_layout —— 把版式展开成每格精确 bbox；读它返回的每格长宽比，据此措辞 prompt
   （宽格写成宽幅建场构图，长格写成特写）。
4. generate_panel —— 每一格各调一次，按 1..N 顺序。
5. compose_page —— 合成整页：分格线、旁白框、气泡台词全部由代码渲染。
6. 最终回复：给出成品页文件路径、格数，以及一句话故事梗概。

风格卡创作方法（默认注入）：
风格卡不是风格名词的堆砌，而是一次微型"艺术运动宣言"：
- 先在心里为本作命名一个视觉运动（1-2 个词，如"粗野主义喜悦"、"午夜银河"），
  style 字段围绕这个运动组织，而不是罗列互不相干的形容词；
- 把运动落实到具体视觉要素：线条、色板、光影、构图、质感，每个要素只说一次，不重复；
- 强调工艺感：画面应像被精心制作的印刷品，而不是随手生成的图片；
- style 控制在 60-120 字中文，是给生图模型的连续指令，不是给人看的散文。

每格 prompt 写法（默认注入，源自社区生图提示词工程）：
按"镜头→主体动作→环境→光色→质量"组织，40-100 字中文。场景密度胜过形容词：
- 镜头先行：先定景别与机位（特写/中景/全景 + 平视/俯仰/倾斜/贴地），再填主体；
- 主体写身体的精确状态："她攥紧车票，指节发白"强过"她很紧张"；
- 环境至少一个可数细节：物件、材质、天气——"房间里"不合格，"榻榻米边缘磨出褐色"合格；
- 光线写方向+色温+光比："右下方矮纸灯，暖黄，光晕极小"；
- 用专业词汇：轮廓光、伦勃朗光、逆光剪影、丁达尔光束、浅景深、贴地平视、斜切构图；
- 正向框架：写想要的替代物，不写"没有/不要X"——负向词会被画面模型吸进画面；
- 抽象情绪必须翻译成可见的视觉隐喻，禁止抽象形容词堆砌。

规则：
- prompt 只描述具体画面；风格卡和角色外貌卡会自动拼进提示词，不要在 prompt 里重复它们。
- 画面中不出现任何文字；所有文字都由代码渲染（旁白框 + 气泡）。
- 漫画节奏：开篇用宽格建场，中间混合大小格推进，把最大或最不规则的一格留给情绪最高点；
  最后一格把寓意落在画面上，不要说破。
- 保持人物一致：远景或人多时，在 prompt 里重复角色的标志性特征。
- 工作流只跑一次。compose_page 成功即任务完成：立刻给出最终回复，
   绝不重启工作流、绝不重复已经成功的工具调用。
- 某一格失败时，把它的 prompt 改得更具体、更简单后重试一次；再失败就跳过并在最终回复中说明。
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 参数列表（测试用）；None 则读 sys.argv。
    Returns:
        带 idea（列表）与 out（输出目录）的 Namespace。
    """
    p = argparse.ArgumentParser(description="把一个想法画成一页漫画")
    p.add_argument("idea", nargs="*", help="你的想法，一句话就行；抽象的更好")
    p.add_argument("--out", default="comic_out", help="输出目录（默认 comic_out）")
    p.add_argument(
        "--debug", action="store_true", help="打印工具的全量参数与返回（不截断）"
    )
    p.add_argument(
        "--style",
        default="auto",
        metavar="SKILL",
        help="画师 skill id；auto=模型按剧本自选（默认）；none=不启用（见 --list-styles）",
    )
    p.add_argument(
        "--list-styles", action="store_true", help="列出可选风格 skill 后退出"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：装配工具与 agent，跑一次完整漫画流水线。

    Args:
        argv: 命令行参数（测试用）。
    Returns:
        进程退出码：0 成功，2 缺少 API key。
    """
    args = parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 — 控制台不支持重配也不影响流程
            pass
    if args.list_styles:
        items = list_skills()
        if not items:
            print("（暂无风格 skill；可往 skills\\ 目录放 .md，或用 Web 上传）")
        for s in items:
            print(f"{s['id']:<16} {s['title']}")
            if s["description"]:
                print(f"{'':<16} {s['description']}")
        return 0
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("IMAGE_API_KEY")):
        print("缺少 API key。最简单：用 .\\run-deepseek.ps1 或 .\\run-glm.ps1 启动。")
        print("手动配置：OPENAI_* / MINIMUSE_MODEL + IMAGE_API_KEY / IMAGE_MODEL")
        return 2

    idea = " ".join(args.idea) or "做一个哲理短篇：关于「追赶昨天的自己」"

    async def _run() -> tuple[str, int]:
        """一次完整生成：先定画师（auto 时模型自选），再跑 agent 工具链。"""
        llm = OpenAILLM()
        style_id = args.style
        if style_id == "auto":  # 索引层交给模型判断；用户显式指定时直接注入
            picked = await choose_skill(idea, llm)
            if picked:
                style_id = picked["id"]
                print(f"[画师自选] {picked['title']}（{picked['id']}）—— {picked['reason']}")
            else:
                style_id = ""
                print("[画师自选] 两次未解析出合法结果，回落为默认风格指引")
        elif style_id == "none":
            style_id = ""
        try:
            system_prompt = apply_skill(COMIC_PROMPT, style_id)
        except ValueError as exc:
            print(exc)
            return "", 2  # 画师 id 写错：在烧图的钱之前就停
        imagegen = OpenAIImageGen()
        sizes = size_map_for(imagegen.model)  # 换画师只换映射表，分镜协议不动
        tools = ToolCollection(
            SaveScript(workspace=out),
            SaveStoryboard(workspace=out),
            PlanLayout(workspace=out, panel_sizes=sizes),
            GeneratePanel(workspace=out, imagegen=imagegen, panel_sizes=sizes),
            ComposePage(workspace=out),
        )
        agent = Agent(
            llm,
            tools,
            max_steps=24,
            system_prompt=system_prompt,
            echo=print,
            debug=args.debug or bool(os.environ.get("MINIMUSE_DEBUG")),
        )
        return await agent.run(idea), 0

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_block = refs_prompt(out)
    if ref_block:  # 有参考图角色：注入剧本第一步，强制使用这些角色
        idea = f"{ref_block}\n\n想法：{idea}"
    answer, code = asyncio.run(_run())
    print("\n" + answer)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

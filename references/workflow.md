# 执行接口

脚本使用 Python 3.12，Pillow、NumPy、OpenCV 的验证版本固定在 `requirements.txt`。无需 OpenAI SDK 或 API Key。首次由 Codex 在仓库根目录执行：

```powershell
./scripts/bootstrap.ps1
# 可显式提供 Python 3.12 路径：./scripts/bootstrap.ps1 -Python C:/Python312/python.exe
```

脚本自动创建 `.venv`、安装固定版本、检查依赖并运行五套离线测试（self-test、test_workflow.py、test_parallel.py、test_portrait.py、test_efficiency.py）；失败立即停止。后续以下命令中的 `python` 均替换为仓库下 `.venv/Scripts/python.exe`。非 Windows 环境用 Python 3.12 执行 `python -m venv .venv`、`.venv/bin/python -m pip install -r requirements.txt` 及这五套测试。固定的是库版本；任务还记录实际 Python/平台/脚本哈希，不承诺跨平台逐字节重算一致。

用户明确要求人像抠图或上半身构图时，按 [人像接口](portrait.md) 设置 portrait=true、选择提取/补全及执行 portrait-layout。人像可在 local-first 下进行必要的 B/extract 或 B/complete；已有透明人像支持 A/native-alpha。该例外不改变普通素材路由。

## 自动发现与计划

```text
python scripts/asset_job.py inventory --input 图片.png 或文件夹 --output 新目录/inventory.json
```

支持多个图片/文件夹输入，文件夹仅扫描当前层。inventory 枚举、生成缩略图并检查 Discovery 缓存，不伪装成视觉识别；缓存未命中由 Codex 看缩略图与必要原图局部，写新的 plan.json。默认使用 [两级分辨率与缓存接口](efficiency.md)，发现完成后 discovery-save 再 build；已有源图坐标计划仍兼容直接 build。计划示例：

```json
{
  "sources": [{"id": "source_001", "path": "C:/images/master.png"}],
  "discovery": {
    "provider": "codex-vision", "status": "complete",
    "coverage_notes": "逐区复查完整主体，保留花叶组合",
    "excluded": ["无独立复用价值的散落小水滴"]
  },
  "candidates": [{
    "id": "source_001_ring", "source_id": "source_001", "label": "青蓝圆环",
    "bbox": [20, 20, 320, 340], "route": "A",
    "reason": "完整圆环与浅色背景可分，保留原始渐变",
    "extraction_method": "matte", "background_rgb": [248, 245, 240],
    "foreground_points": [[100, 100]], "background_points": [[170, 170]]
  }]
}
```

示例坐标仅说明格式，实际 bbox、背景采样及前景/背景点由 Codex 根据原图确定。大模型负责识别类别、定位与参数；Python 根据这些参数分离原像素、生成 Alpha、预览和 PNG，不调用通用语义抠图模型。

ID 仅 ASCII 字母、数字、下划线、连字符且唯一。bbox=[left,top,right,bottom]，右下不包含，基于 EXIF 校正后源图；点坐标也使用完整源图坐标。route 接受 A/B/C 或旧 AUTO/IMAGE2/MANUAL，存储兼容旧名称。B 必填具体 repair_prompt；repair_mode 为 extract 或 complete。repair_allowed=false 禁止生成式回退和补全。

重叠图形计划只分「组合素材」与「拆解素材」：先按可复用区域建立组合候选（例如 `red_circle_lines_combo`），bbox 包住完整组合，reason 说明包含的图形与交叠关系；原图中完整、且能干净去掉相邻碎片的主体可另建拆解候选。若候选预览混入其他图形残片，修改 bbox/前景点后重建新任务，或按实际范围改为组合；不可把混入碎片的结果当作拆解素材。对被遮挡的不可见部分默认不建独立补全任务；只有用户明确要求时才另建 B/complete，并标明生成式来源。最终清单按组合素材与拆解素材分组报告。

A/matte 必填实际采样 background_rgb，可选 background_points 标记真实内孔，可选 foreground_points 选择目标连通组件。脚本保留不与背景连通的内部浅色内容。颜色阈值仅为平底启发式，不代表质量评分。可本地处理的候选写 A/AUTO，并提供已支持的方法及必需参数；不能将缺参数的 B/extract 当成自动本地抠图入口。

A 的默认 extraction_method=matte 使用上述颜色参数，先检查未加边距的提取结果，再添加透明外边距；padding 默认 12，matte 可设 0，其余方法为 1..512 的整数。另支持 `crop`（完整矩形照片裁切）和 `bright-background`（浅色背景上的深色不透明主体，用 GrabCut 分割并向内软化边缘）；后两者不需要 background_rgb。crop 保留照片内部背景，仅外围透明，不能用来掩盖主体裁断或抠图失败。bright-background 不是通用语义分割，对浅色主体/高光/玻璃不适用，必须检查深浅底，处理失败或视觉拒绝均转人工。若源是模型生成回图，候选必须标记 generated_source=true，并在 reason 中关联原任务/尝试。所有本地输出都先 REVIEW，不能自动 PASS。

```text
python scripts/asset_job.py build --plan plan.json --job 新任务目录 --workers 5 --processing local-first
python scripts/asset_job.py review --job 任务目录 --id source_001_ring --decision accept --note "对照原图与深浅底：主体完整，内孔透明，无背景残留"
```

build 不调用内置工具。`--workers` 接受正整数，默认 5，新任务在 manifest 保存 `max_parallel`；本地候选由线程池处理，实际线程数不超过候选数量，主线程按原计划顺序汇总并保存任务记录。每次新建任务避免覆盖旧来源。计划参数无效时拒绝建任务；非处理错误仍为 ERROR，不混同人工判断。

`--processing` 默认 `local-first`，记录在 manifest：对普通非 complete 候选设置 repair_allowed=false，普通提取或复核失败转人工，不调用图片模型；未提供本地提取指导的普通 B/extract 保存为人工任务并说明原因。B/complete 仅在用户已明确要求遮挡补全时建立；明确人像请求则按 portrait.md 执行必要的分离/补全，两者都不能覆盖 repair_allowed=false。另一选项 `builtin-repair` 仅在用户明确选择一般生成式分离/修补路线时使用，恢复历史 A 失败可进 B、B/extract 可调用工具的行为；普通 bright-background 失败仍转人工。旧任务没有 processing 字段时不自动迁移或改变行为。

## 并发调度

主 agent 在发现阶段按 source ID 分工。给每个 worker 独立的源图路径、实际尺寸、候选 ID 前缀及计划片段输出路径；worker 只写自己的片段。主 agent 合并为一个 plan.json，检查来源、重复候选及整体覆盖后统一 build。单张原图直接发现，不为这一项创建 subagent；该图发现出的多个独立素材仍可在后续并行处理。

本地处理统一交给 build 的线程池。subagent 主要承担发现和复核；A 快速通道按互斥 Contact Sheet 批次分工，其余按互斥素材 ID 分工，每个 worker 收到任务目录绝对路径、批次索引/素材 ID、原裁切和实际预览路径。默认继承主 agent 的模型，一次分配只交给一个 worker，完成一项立即补充下一项。A 输出经 Alpha 检查后仍必须看原图、深浅底再审核，可用 [review-sheet/review-batch](efficiency.md) 批量复核，存疑项 detail 后单独细看；C 保留人工原因，主 agent 汇总并 finalize/verify。仅必要且已授权的生成式任务由 worker 完成查看裁切、登记、调用、导入及复核；同一素材保持顺序。

本地线程池按候选数量和 workers 上限调度，不受 agent 名额直接限制；配置 5 时可并行提取 5 项。发现和复核按独立任务数及宿主实际 agent 名额调度。生成式补全另受图片工具并发限制，不能为了凑足并行数量把本地候选送去生成。宿主没有 subagent 时由主 agent 发现和复核，脚本仍按配置批量处理；保留限制与实际耗时说明。

repair-queue 与 status 返回 `capacity`（`max_parallel`、`active`、`available_slots`）及 `active_tasks`。REPAIRING 和 REPAIR_BLOCKED 均计入占用名额；未记录 `max_parallel` 的旧任务按单路处理，不自动改写并发配置。队列可返回全部等待候选，主 agent 只向空闲名额分发；可用名额为零时推进已启动任务的结果恢复或复核。

repair-start 在文件锁内读取最新状态、检查名额和两次请求上限、保存本次尝试，成功后才允许工具调用；`--worker` 可记录 agent 标识。多个 worker 同时抢到同一素材或超出名额时，未成功者不能调用工具，重新读取队列继续调度。读取和修改共享记录的命令均使用任务锁；图片生成在锁外执行，避免把外部等待串行化。worker 不直接编辑 manifest、plan 或共享预览。运行时锁文件不计入交付校验。

## 内置工具调用与结果导入

本节仅用于用户明确要求的 B/complete、人像分离/补全，或明确选择 builtin-repair 的任务。local-first 普通素材提取失败不进入此队列。

例如用户要求补全被叶片遮挡的贝壳，额外建立 `source_001_shell_complete`，设置 route=B、repair_mode=complete 和具体 repair_prompt；仍按计划的 repair_allowed 执行。不要把“主体被遮挡”自动理解为补全授权。

repair-queue 返回候选参考图绝对路径和 prompt。repair-start 返回 `id`、`attempt`、本次固定 `prompt` 及 `referenced_image_paths`，并将状态置为 REPAIRING。worker 先用 view_image 查看明确的候选裁切，再按返回路径通过 image_gen 的 `referenced_image_paths` 调用一次；当前工具 schema 为准，不添加 model 等未开放参数。不能用 `num_last_images_to_include` 或回图完成顺序绑定素材，尤其不能把另一 worker 的最近图像带入调用。

```text
python scripts/asset_job.py repair-queue --job 任务目录
python scripts/asset_job.py repair-start --job 任务目录 --id source_001_shell_complete --worker agent-shell
python scripts/asset_job.py repair-result --job 任务目录 --id source_001_shell_complete --attempt 1 --input 工具实际返回文件.png
python scripts/asset_job.py review --job 任务目录 --id source_001_shell_complete --decision accept --note "原可见部分保持，缺失区域为生成式补全；对照深浅底完整透明，无背景残留"
```

新任务必须带 repair-start 返回的 `--attempt N`；导入图片和记录失败均核对当前尝试，迟到的其他编号被拒绝，不能覆盖新一轮回图或标记新一轮失败。未记录 `max_parallel` 的旧任务允许省略编号以保持兼容。结果关联使用任务目录、素材 ID、尝试编号和工具返回的实际文件路径，与完成顺序无关。

仅工具实际返回模型信息时加 --model；工具结果 ID 可用 --tool-reference 记录。图片先复制到 repaired/id-attempt-N.png，透明检查通过后写 review、masks、previews，等视觉复核放行。已有 Alpha 原样保留。没有真实透明、主体被裁切或全透明均不通过；第一次回到队列，第二次转人工。

视觉不合格用 review --decision reject --note 具体问题；保留拒绝文件，允许生成式修补时最多再修一次，严格模式及 bright-background 拒绝后转人工。审核前先保存 decision、note 和目标路径；若移文件或提交状态时中断，status 会返回 pending_review，用同一候选及已记录的 decision、note 再运行 review，不能改成另一种决定。已完成的候选无需再审核。某次工具调用中断或返回不明：

```text
python scripts/asset_job.py repair-result --job 任务目录 --id source_001_shell_complete --attempt 1 --failure "工具超时，是否生成未知"
```

置 REPAIR_BLOCKED，退出码 2，继续占用名额。先查原调用结果，找到真实文件可对同次请求重新 repair-result --attempt N --input，不发送重复请求；只使用其余可用名额处理其他素材。本版没有自动恢复一个未知外部调用的接口。

旧 repaired 子命令仅供未记录 `max_parallel` 的历史任务手动导入网页回图，要求 --background。新任务拒绝此入口，必须先 repair-start，再用 repair-result --attempt N 导入，以免绕过尝试记录或破坏原生透明通道。

## 记录与检查

manifest 的 repair_attempts 保存每次 prompt、provider、model、worker、工具引用、来源和 SHA-256。`started_at` 是登记尝试的本地时间，`received_at` 是本地收到回图的时间，都不是 provider 内部生成的开始或完成时间。真实并发验证需另保存工具调用起止时间及事件，报告实际调用重叠数量、总耗时与通过数量；不能用登记时间或配置上限代替并发证据，也不预先承诺固定倍数提速。

model=null 表示工具没有明确披露，不代表特定 Images 型号。generated_repair=true 表示图像模型处理，哪怕是 extract 也不能说像素未变。

状态：REVIEW、PASS、WAITING_REPAIR、REPAIRING、REPAIR_BLOCKED、MANUAL、REJECTED、ERROR。只有 PASS 进入 assets。旧诊断任务不会被自动迁移或改写。

## 自动推进与交付

```text
python scripts/asset_job.py status --job 任务目录
python scripts/asset_job.py finalize --job 任务目录
python scripts/asset_job.py verify --job 任务目录
```

status 返回每个未解决候选的下一步动作及原图/预览路径。Codex 循环执行动作直到 PASS/MANUAL 或明确外部阻塞。finalize 拒绝任何未解决候选，校验 PASS 文件，并检查人工任务的 PNG、Markdown、JSON 是否齐全及任务图是否与原候选一致，输出 `delivery.json`（本地通过、生成通过、人工数量及文件路径）和 `checksums.json`。整个目录包含计划、源图快照、候选、每次回图、审核及人工任务，可整体复制后 verify。verify 检查清单文件是否缺失或改变；这不是防恶意修改的签名，也不重新证明视觉质量。任务被修改后需重新 finalize。

导入在保存回图后中断，可对同一次请求再次 repair-result；已有回图必须与传入图片像素一致，不覆盖另一张图。B + repair_allowed=false、A 严格模式质量失败、修补两次失败均转人工并生成图和原因。人工处理的完成不由脚本假定。

可复现离线样例（合成图，未调用图片模型）：

```text
python scripts/test_workflow.py --output outputs/offline-demo
python scripts/test_parallel.py --output outputs/parallel-demo
python scripts/asset_job.py verify --job outputs/offline-demo/synthetic-workflow
```

样例验证导入、恢复、人工收尾和交付校验；并发样例检查容量、竞争、乱序结果关联及顺序重试。审核注释明确标记 synthetic，不能当作真实图片模型验收。输出目录必须尚未使用。

```text
python scripts/asset_job.py self-test
python scripts/test_workflow.py
python scripts/test_parallel.py
```

自检验证程序与状态，不证明模型质量。真实调用必须另查工具结果和输出图像。`/extract-assets` 未注册；用户直接用自然语言或 `$design-asset-extractor`。

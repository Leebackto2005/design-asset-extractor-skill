# 执行接口

脚本使用 Python 3.12，Pillow、NumPy、OpenCV 的验证版本固定在 `requirements.txt`。无需 OpenAI SDK 或 API Key。首次由 Codex 在仓库根目录执行：

```powershell
./scripts/bootstrap.ps1
# 可显式提供 Python 3.12 路径：./scripts/bootstrap.ps1 -Python C:/Python312/python.exe
```

脚本自动创建 `.venv`、安装固定版本、检查依赖并运行两套离线测试；失败立即停止。后续以下命令中的 `python` 均替换为仓库下 `.venv/Scripts/python.exe`。非 Windows 环境用 Python 3.12 执行 `python -m venv .venv`、`.venv/bin/python -m pip install -r requirements.txt` 及两套测试。固定的是库版本；任务还记录实际 Python/平台/脚本哈希，不承诺跨平台逐字节重算一致。

## 自动发现与计划

```text
python asset_job.py inventory --input 图片.png 或文件夹 --output 新目录/inventory.json
```

支持多个图片/文件夹输入，文件夹仅扫描当前层。inventory 只枚举，不伪装成视觉发现；Codex 随后逐张查看 source 原图，自动写新的 plan.json：

```json
{
  "sources": [{"id": "source_001", "path": "C:/images/master.png"}],
  "discovery": {
    "provider": "codex-vision", "status": "complete",
    "coverage_notes": "逐区复查完整主体，保留花叶组合",
    "excluded": ["无独立复用价值的散落小水滴"]
  },
  "candidates": [{
    "id": "source_001_turtle", "source_id": "source_001", "label": "海龟",
    "bbox": [380, 340, 790, 700], "route": "B",
    "reason": "主体完整，渐变天空不适合平底法", "repair_mode": "extract",
    "repair_prompt": "仅分离海龟，保持姿态、龟壳纹路、鳍和头部，移除周围背景及其他元素。"
  }]
}
```

ID 仅 ASCII 字母、数字、下划线、连字符且唯一。bbox=[left,top,right,bottom]，右下不包含，基于 EXIF 校正后源图。route 接受 A/B/C 或旧 AUTO/IMAGE2/MANUAL，存储兼容旧名称。B 必填具体 repair_prompt；repair_mode 为 extract 或 complete。repair_allowed=false 禁止失败后生成式回退。

重叠图形计划只分「组合素材」与「拆解素材」：先按可复用区域建立组合候选（例如 `red_circle_lines_combo`），bbox 包住完整组合，reason 说明包含的图形与交叠关系；原图中完整、且能干净去掉相邻碎片的主体可另建拆解候选。若候选预览混入其他图形残片，修改 bbox/前景点后重建新任务，或按实际范围改为组合；不可把混入碎片的结果当作拆解素材。对被遮挡的不可见部分默认不建独立补全任务；只有用户明确要求时才另建 B/complete，并标明生成式来源。最终清单按组合素材与拆解素材分组报告。

A 必填采样 background_rgb，可选 background_points 标记真实内孔，可选 foreground_points 选择目标连通组件。脚本保留不与背景连通的内部浅色内容。颜色阈值仅为平底启发式，不代表质量评分。

A 的默认 extraction_method=matte 使用上述颜色参数。另支持 `crop`（精确矩形裁切，padding 默认 12 像素）和 `bright-background`（浅色背景上的深色不透明主体，用 GrabCut 分割并向内软化边缘）；后两者不需要 background_rgb。crop 保留照片内部背景，仅外围透明。bright-background 不是通用语义分割，对浅色主体/高光/玻璃不适用，必须检查深浅底。若源是模型生成回图，候选必须标记 generated_source=true，并在 reason 中关联原任务/尝试。两种本地输出都先 REVIEW，不能自动 PASS。

```text
python asset_job.py build --plan plan.json --job 新任务目录
python asset_job.py repair-queue --job 任务目录
python asset_job.py repair-start --job 任务目录 --id source_001_turtle
```

build 不调用内置工具；队列由 Codex 继续执行。每次新建任务避免覆盖旧来源。A 数据检查失败会带原因进入 B；非处理错误仍为 ERROR，不混同人工判断。

## 内置工具调用与结果导入

repair-queue 返回候选参考图绝对路径和 prompt。repair-start 返回本次固定提示词并将状态置为 REPAIRING。随后 Codex 用 image_gen 调用一次，当前工具 schema 为准，不添加 model 等未开放参数。

```text
python asset_job.py repair-result --job 任务目录 --id source_001_turtle --input 工具实际返回文件.png
python asset_job.py review --job 任务目录 --id source_001_turtle --decision accept --note "对照原图及深浅底：主体完整，风格保持，无背景残留"
```

仅工具实际返回模型信息时加 --model；工具结果 ID 可用 --tool-reference 记录。图片先复制到 repaired/id-attempt-N.png，透明检查通过后写 review、masks、previews，等视觉复核放行。已有 Alpha 原样保留。没有真实透明、主体被裁切或全透明均不通过；第一次回到队列，第二次转人工。

视觉不合格用 review --decision reject --note 具体问题；保留拒绝文件，最多再修一次。某次调用中断或返回不明：

```text
python asset_job.py repair-result --job 任务目录 --id source_001_turtle --failure "工具超时，是否生成未知"
```

置 REPAIR_BLOCKED，退出码 2。先查原调用结果，找到真实文件可对同次请求重新 repair-result --input，不发送重复请求。本版没有自动恢复一个未知外部调用的接口。

旧 repaired 子命令保留为手动网页回图兼容入口，要求 --background；新的内置路线用 repair-result，以免破坏原生透明通道。

## 记录与检查

manifest 的 repair_attempts 保存每次 prompt、provider、model、工具引用、来源和 SHA-256。model=null 表示工具没有明确披露，不代表特定 Images 型号。generated_repair=true 表示图像模型处理，哪怕是 extract 也不能说像素未变。

状态：REVIEW、PASS、WAITING_REPAIR、REPAIRING、REPAIR_BLOCKED、MANUAL、REJECTED、ERROR。只有 PASS 进入 assets。旧诊断任务不会被自动迁移或改写。

## 自动推进与交付

```text
python scripts/asset_job.py status --job 任务目录
python scripts/asset_job.py finalize --job 任务目录
python scripts/asset_job.py verify --job 任务目录
```

status 返回每个未解决候选的下一步动作及原图/预览路径。Codex 循环执行动作直到 PASS/MANUAL 或明确外部阻塞。finalize 拒绝任何未解决候选，校验 PASS 文件，输出 `delivery.json`（本地通过、生成通过、人工数量及文件路径）和 `checksums.json`。整个目录包含计划、源图快照、候选、每次回图、审核及人工任务，可整体复制后 verify。verify 检查清单文件是否缺失或改变；这不是防恶意修改的签名，也不重新证明视觉质量。任务被修改后需重新 finalize。

导入在保存回图后中断，可对同一次请求再次 repair-result；已有回图必须与传入图片像素一致，不覆盖另一张图。B + repair_allowed=false、A 严格模式质量失败、修补两次失败均转人工并生成图和原因。人工处理的完成不由脚本假定。

可复现离线样例（合成图，未调用图片模型）：

```text
python scripts/test_workflow.py --output outputs/offline-demo
python scripts/asset_job.py verify --job outputs/offline-demo/synthetic-workflow
```

样例验证导入、恢复、人工收尾和交付校验；审核注释明确标记 synthetic，不能当作真实图片模型验收。输出目录必须尚未使用。

```text
python asset_job.py self-test
python scripts/test_workflow.py
```

自检验证程序与状态，不证明模型质量。真实调用必须另查工具结果和输出图像。`/extract-assets` 未注册；用户直接用自然语言或 `$design-asset-extractor`。

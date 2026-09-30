# 快速通道与发现缓存

这是现有工作流的效率接口，沿用 [workflow.md](workflow.md) 的路由、锁、修补次数与交付验收。用户只需给图片与处理目标；以下文件与参数由 Codex 建立。

## 两级分辨率与 Discovery 缓存

```text
python scripts/asset_job.py inventory --input 原图.png --output inventory.json --proxy-size 1024 --intent "提取几何组合与独立主体，保留原像素，不补全"
```

inventory 枚举原图、读取 EXIF 校正后的尺寸和 SHA-256，生成长边不超过 1024 的缩略图；--proxy-size 可设 256..2048，不放大小图。脚本不进行语义识别。输出 sources 的 proxy_path、proxy_size、proxy_to_source 供模型发现和定位；精细结构需看原图局部。原图作为第二级，最终提取从原图像素进行，不使用放大的缩略图蒙版，也不对每个素材先低分辨率抠图再重做一次。

缓存默认在用户目录 `.cache/design-asset-extractor/discovery`，可用 --cache-dir 指定。键包括原图内容哈希、尺寸与 source 顺序、真实任务意图、缩略图尺寸、发现格式版本和脚本哈希。任何代码更新都会保守地失效。--intent 要包含用户选择与约束；同图改成人像、补全、保留组合等不同要求，必须改变 intent。--refresh-discovery 强制重新视觉发现，但完整的缩略图可以复用。缓存只保存候选计划，不存审核许可。

未命中时 Codex 看缩略图与必要局部，写 discovery.status=complete、coverage_notes、excluded 及候选计划。候选格式沿用 workflow.md。所有候选和点必须使用同一坐标系：

```json
{
  "coordinate_space": "proxy",
  "sources": [{"id": "source_001", "path": "C:/images/original.png"}],
  "discovery": {"provider": "codex-vision", "status": "complete", "coverage_notes": "已复查主体与组合"},
  "candidates": [{
    "id": "ring", "source_id": "source_001", "label": "圆环", "route": "A",
    "reason": "完整主体与平底可分", "bbox": [20, 20, 200, 250],
    "background_rgb": [248, 245, 238], "background_points": [[100, 120]]
  }]
}
```

坐标仅示意，实际由看图确定。默认 coordinate_space=source，可使用原图坐标。转换时 bbox 左上取 floor、右下取 ceil，点按实际原图/缩略图比例取整。保存命令核对来源、源文件哈希、缩略图哈希、计划完整度和坐标：

```text
python scripts/asset_job.py discovery-save --inventory inventory.json --plan model-plan.json --output ready-plan.json
python scripts/asset_job.py build --plan ready-plan.json --job 新任务目录 --workers 5
```

再次 inventory 返回 cache_hit=true 时，inventory.json 本身包含源图坐标的完整计划，可核对意图和覆盖后直接 build；仍为新任务，仍先 REVIEW。用户要求重新发现或缓存不适合当前意图时，不直接使用它。build 检查规范计划中的源哈希；源图变化拒绝建任务，重新 inventory/发现。直接写原有 plan/build 仍兼容，只是不使用发现缓存。

## 自动 QC 后批量看图

符合快速通道的条件：普通 A/AUTO、matte 或 crop、自动 QC 通过、前景包围框宽高至少 24 像素、具备当前预览哈希，且不是人像、生成式输出、待恢复事务或明确存疑项。24 像素是转单项细看的阈值，不是质量评分；大素材的细边仍可能需要放大。bright-background、生成式分离/补全与人像默认单项复核，普通候选可设置 requires_detail_review=true。

```text
python scripts/asset_job.py review-sheet --job 任务目录 --batch-size 8
```

返回 batches（index、index_sha256、image、ids）与 detail_ids。每张最多 8 行，--batch-size 支持 1..12。每行保留原有 900×350 原裁切/浅底/深底预览，不再把它缩小到半宽。Codex 或负责该批的 worker 必须查看 image 实际图像，对每个 ID 分别判断；看不清就 detail。不能只看数量、读取 JSON 或替用户制造批量通过记录。

看过图后由模型写决定文件，使用返回的 index_sha256；decisions 必须完整覆盖该索引所有 ID 一次，每项有具体观察：

```json
{
  "index_sha256": "使用实际返回值",
  "decisions": [
    {"id": "ring", "decision": "accept", "note": "环体完整，内孔随深浅底变化，无背景块或邻物残留"},
    {"id": "leaf", "decision": "detail", "note": "右侧细叶缘在本拼版中无法辨认，需放大检查"}
  ]
}
```

示例 ID 以实际索引为准；decision=accept/reject/detail（亦接受 PASS/REJECT/DETAIL）。

```text
python scripts/asset_job.py review-batch --job 任务目录 --sheet 返回的索引绝对路径.json --decisions decisions.json
```

accept 经完整检查才移动到 assets；reject 沿用原路由和生成权限，local-first 普通素材转人工；detail 保持 REVIEW，标记需单独检查，然后用原 review 命令审核。没有“自动接受整批”或抽样放行。

索引绑定每个候选、审核图、预览与拼版的哈希。素材/拼版改变、错用索引、缺项、重复 ID 或空观察会拒绝提交。在锁内先验证整批，再记录所有待提交决定，最后移动文件并保存一次完成状态。中断后重放相同索引与原 decisions；已经完成且决定一致的行可以安全重放。status 的 pending_review 也允许按原 note 单项恢复；不要改变尚待恢复的决定。

批次是互斥调度单位：一个批次只由一个 worker 负责，不同时分发其中 ID 做单项审核。存疑项升级后再单独分配。沿用全局任务锁，未新增每素材状态文件或独立锁。

## 解码与哈希复用边界

build 在总计 256 MiB RGBA 预算内把每张原图解码一次，供该来源的线程只读裁切；超出预算的来源回退磁盘读取。实际线程数仍为 min(workers, 候选数)，PNG 与最终蒙版保持原分辨率。

SHA-256 流式读取；同一个命令内，相同规范路径及未改变的文件 stat 可以复用结果。文件写入/替换后重新算，命令退出清空缓存。不以文件名、旧时间戳或永久缓存代替完整性检查。finalize 验证与索引建立可复用已检查的同一文件哈希；下一次命令重新读取，verify 强制重新算每个交付文件，运行时锁不进入清单。

效率验收依据是实际重复读取减少、缓存命中与结果一致，不能根据机制推断固定倍数。计时需区分脚本与模型识别/看图/工具等待；见 [实测记录](../docs/efficiency-validation-2026-10-01.md)。

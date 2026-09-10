# 计划和命令

脚本为 `scripts/asset_job.py`，路径需按当前 Skill 安装位置展开并加引号。

```json
{
  "sources": [{"id": "sheet_01", "path": "C:/images/sheet.png"}],
  "candidates": [{
    "id": "sheet_01_ring", "source_id": "sheet_01", "label": "渐变圆环",
    "bbox": [500, 970, 805, 1300], "route": "AUTO",
    "reason": "不透明独立圆环，米白平底",
    "background_rgb": [248, 245, 238],
    "background_points": [[645, 1135]]
  }]
}
```

- ID 仅 ASCII 字母、数字、下划线、连字符，任务内唯一；label 可中文。计划包含每张源图，脚本逐个复制为 PNG 并记录原路径、SHA-256。
- bbox 为源图像素 `[left,top,right,bottom]`，右/下边界不包含。
- AUTO 必填实际采样的 background_rgb。background_points 使用源图坐标，标记与外围不连通但确实应透明的空隙；不要指向卡片内容或高光。
- 颜色距离是 RGB 欧氏距离，8 以下透明，36 以上不作为背景。边缘结合邻近不透明前景估计覆盖率并去背景色；只是均匀背景启发式，不是质量分。
- 可选 foreground_points 使用源图坐标标记要保留的前景连通区域，排除裁切框内旁边的素材；多组件组合每个组件都要标记。不提供时保留所有前景。不能分开已经粘连的对象。
- IMAGE2 必填具体 repair_prompt 和 reason。例如“移除前方彩带，补全黄色太阳被遮挡部分，保持已有风格、颜色、轮廓，不新增元素，使用均匀背景并留边”。MANUAL 只需原因。

```text
python asset_job.py build --plan plan.json --job 新任务目录
python asset_job.py review --job 任务目录 --id sheet_01_ring --decision accept --note "已查看深浅底，轮廓和空隙完整，无可见残留"
python asset_job.py review --job 任务目录 --id sheet_01_ring --decision reject --note "边缘残留背景色"
python asset_job.py repaired --job 任务目录 --id sheet_01_sun --input 修补图.png --background 248 245 238
python asset_job.py self-test
```

review 的 --id 可传多个，但仅对实际逐一查看的素材使用。repaired 用重复的 `--point x y` 标记回图空隙（回图坐标）。本版只接收首次回图，第二轮另建任务，避免覆盖历史。

目录：source 原图副本、candidates 裁切、masks Alpha、review 待验收 PNG、previews 三联图和总览、assets 通过、review_image2 修补包、manual 人工任务、rejected 拒绝、repaired 回图，以及 manifest.json。

状态：REVIEW、PASS、WAITING_REPAIR、MANUAL、REJECTED、ERROR。只有 PASS 计入可用输出。build 单个候选处理失败会记录并继续，最终退出码非零；脚本不会自己发现或语义分流候选。

使用自然语言或 `$design-asset-extractor` 调用；方案中的 `/extract-assets` 不是已注册命令。

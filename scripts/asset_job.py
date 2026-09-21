"""Candidate discovery handoff, A/B/C routing, built-in image repair and review."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import shutil
import platform
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import PIL


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'pillow': PIL.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
            'script_sha256': digest(__file__)}


def manual(job, asset, reason):
    """All routes use the same handoff; retain previous attempts and reviews."""
    aid = asset['id']
    shutil.copyfile(job / 'candidates' / f'{aid}.png', job / 'manual' / f'{aid}.png')
    asset.update(status='MANUAL', manual_reason=reason)
    (job / 'manual' / f'{aid}.json').write_text(
        json.dumps(asset, ensure_ascii=False, indent=2), encoding='utf-8')
    (job / 'manual' / f'{aid}.md').write_text(
        f"# {asset['label']} ({aid})\n\n{reason}\n\n"
        f"Source: ../source/{asset['source_id']}.png\n\n"
        f"Crop: {aid}.png\n\nAttempts and review: {aid}.json\n", encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGBA')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
        raise ValueError('Unsafe or missing ID')
    return value


def save_manifest(job, manifest):
    manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
    manifest['counts'] = {s: sum(a['status'] == s for a in manifest['assets'])
                          for s in ['REVIEW', 'PASS', 'WAITING_REPAIR', 'REPAIRING', 'REPAIR_BLOCKED', 'MANUAL', 'REJECTED', 'ERROR']}
    tmp = job / 'manifest.tmp'
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(job / 'manifest.json')


def load_job(path):
    job = Path(path).resolve()
    manifest = json.loads((job / 'manifest.json').read_text(encoding='utf-8'))
    for asset in manifest['assets']:
        identifier(asset['id'])
    return job, manifest


def color(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or
            any(type(c) is not int or not 0 <= c <= 255 for c in value)):
        raise ValueError('background_rgb must contain three integers in 0..255')
    return np.array(value, dtype=np.float32)


def matte(im, background, points=(), foreground_points=(), tolerance=36):
    # ponytail: uniform backgrounds only; semantic segmentation is a future need.
    if type(tolerance) is not int or not 16 <= tolerance <= 120:
        raise ValueError('background_tolerance must be an integer in 16..120')
    im = im.convert('RGBA')
    rgb = np.asarray(im.convert('RGB'), dtype=np.float32)
    bg = color(background)
    distance = np.linalg.norm(rgb - bg, axis=2)
    _, labels = cv2.connectedComponents((distance < tolerance).astype(np.uint8), connectivity=8)
    exterior = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    selected = set(exterior.tolist()) - {0}
    for x, y in points:
        if not (0 <= x < im.width and 0 <= y < im.height) or distance[y, x] > tolerance * 0.45:
            raise ValueError('Background point outside crop or not close to background color')
        selected.add(int(labels[y, x]))
    connected = np.isin(labels, list(selected))
    alpha = np.where(connected, np.clip((distance - tolerance * 0.22) / (tolerance * 0.78), 0, 1), 1)
    clean = np.clip((rgb - (1 - alpha[..., None]) * bg) / np.maximum(alpha[..., None], 1 / 255), 0, 255)
    alpha *= np.asarray(im.getchannel('A'), dtype=np.float32) / 255
    rgba = np.dstack([np.rint(clean).astype(np.uint8), np.rint(alpha * 255).astype(np.uint8)])
    if foreground_points:
        _, components = cv2.connectedComponents((rgba[:, :, 3] > 0).astype(np.uint8), connectivity=8)
        keep = set()
        for x, y in foreground_points:
            if not (0 <= x < im.width and 0 <= y < im.height) or rgba[y, x, 3] < 200:
                raise ValueError('Foreground point must lie inside an opaque target')
            keep.add(int(components[y, x]))
        rgba[~np.isin(components, list(keep)), 3] = 0
    # Estimate edge coverage from nearby opaque color, not RGB distance alone.
    # Otherwise antialiased dark lines retain a light backdrop-colored fringe.
    core = cv2.erode((rgba[:, :, 3] == 255).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if core.any():
        _, nearest = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros((int(nearest.max()) + 1, 3), dtype=np.float32)
        lookup[nearest[core]] = rgb[core]
        fg = lookup[nearest]
        direction = fg - bg
        coverage = np.clip(np.sum((rgb - bg) * direction, axis=2) /
                           np.maximum(np.sum(direction * direction, axis=2), 1), 0, 1)
        edge = (rgba[:, :, 3] > 0) & ~core
        coverage *= np.asarray(im.getchannel('A'), dtype=np.float32) / 255
        rgba[edge, 3] = np.rint(coverage[edge] * 255).astype(np.uint8)
        unmixed = np.clip((rgb - (1 - coverage[..., None]) * bg) /
                         np.maximum(coverage[..., None], 1 / 255), 0, 255)
        rgba[edge, :3] = np.rint(unmixed[edge]).astype(np.uint8)
    rgba[rgba[:, :, 3] == 0, :3] = 0
    return Image.fromarray(rgba)


def inspect(im):
    alpha = np.asarray(im.getchannel('A'))
    foreground = alpha > 8
    if not foreground.any() or not (alpha == 0).any():
        raise ValueError('Empty foreground or no fully transparent background')
    if (alpha[0] > 8).any() or (alpha[-1] > 8).any() or (alpha[:, 0] > 8).any() or (alpha[:, -1] > 8).any():
        raise ValueError('Foreground touches crop edge: expand crop or route to manual')
    ys, xs = np.where(foreground)
    return {'width': im.width, 'height': im.height,
            'foreground_bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            'foreground_fraction': round(float(foreground.mean()), 4),
            'transparent_pixels': int((alpha == 0).sum()),
            'soft_pixels': int(((alpha > 0) & (alpha < 255)).sum())}


def local_extract(im, method, padding=12):
    if type(padding) is not int or not 1 <= padding <= 512:
        raise ValueError('padding must be an integer in 1..512')
    if method == 'crop':
        output = Image.new('RGBA', (im.width + 2 * padding, im.height + 2 * padding))
        output.paste(im, (padding, padding))
        return output
    if method != 'bright-background':
        raise ValueError('Unsupported local extraction method')
    # Bounded fallback for dark opaque subjects on light neutral backdrops.
    # Not a general semantic segmenter: white/transparent subjects require manual work.
    rgb = np.asarray(im.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if np.percentile(border, 5) < 175:
        raise ValueError('Bright-background method requires a clear light border')
    mask = np.where(gray < 160, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[gray > 195] = cv2.GC_BGD
    mask[gray < 70] = cv2.GC_FGD
    mask[[0, -1], :] = cv2.GC_BGD
    mask[:, [0, -1]] = cv2.GC_BGD
    if not (mask == cv2.GC_FGD).any():
        raise ValueError('No reliable dark foreground seed')
    cv2.setRNGSeed(0)
    cv2.grabCut(rgb, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    # Feather inward only: do not reintroduce bright backdrop into edge RGB.
    inside = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    alpha = np.rint(np.clip(inside / 2, 0, 1) * 255).astype(np.uint8)
    alpha = np.minimum(alpha, np.asarray(im.getchannel('A')))
    rgba = np.dstack([rgb, alpha])
    core = inside >= 8
    if core.any():
        _, nearest = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        colors = np.zeros((int(nearest.max()) + 1, 3), dtype=np.uint8)
        colors[nearest[core]] = rgb[core]
        edge = (alpha > 0) & ~core
        rgba[edge, :3] = colors[nearest[edge]]
    rgba[alpha == 0, :3] = 0
    result = Image.fromarray(rgba)
    inspect(result)  # Check before padding so a clipped subject cannot pass.
    return result


def preview(source, output, target):
    canvas = Image.new('RGB', (900, 350), '#dddddd')
    draw = ImageDraw.Draw(canvas)
    for i, (label, backdrop) in enumerate([('SOURCE', '#ffffff'), ('LIGHT', '#f4f4f4'), ('DARK', '#242932')]):
        item = source if i == 0 else output
        base = Image.new('RGBA', item.size, backdrop)
        base.alpha_composite(item)
        small = ImageOps.contain(base.convert('RGB'), (300, 320))
        canvas.paste(small, (i * 300 + (300 - small.width) // 2, 25 + (320 - small.height) // 2))
        draw.text((i * 300 + 10, 7), label, fill='black')
    canvas.save(target)


def process(job, asset, im, background=None, points=(), foreground_points=()):
    method = asset.get('extraction_method', 'matte')
    if method == 'matte' and asset.get('padding', 12):
        padding = asset.get('padding', 12)
        if type(padding) is not int or not 1 <= padding <= 512:
            raise ValueError('padding must be an integer in 1..512')
        im = ImageOps.expand(im, border=padding, fill=tuple(background))
        points = [(x + padding, y + padding) for x, y in points]
        foreground_points = [(x + padding, y + padding) for x, y in foreground_points]
    output = (matte(im, background, points, foreground_points, asset.get('background_tolerance', 36)) if method == 'matte'
              else local_extract(im, method, asset.get('padding', 12)))
    metrics = inspect(output)
    aid = asset['id']
    output.getchannel('A').save(job / 'masks' / f'{aid}.png')
    output.save(job / 'review' / f'{aid}.png')
    preview(im, output, job / 'previews' / f'{aid}.jpg')
    asset.update(status='REVIEW', metrics=metrics, review_sha256=digest(job / 'review' / f'{aid}.png'))


def contact_sheet(job, manifest):
    files = [(a['id'], job / 'previews' / (a['id'] + '.jpg')) for a in manifest['assets']]
    files = [(aid, path) for aid, path in files if path.exists()]
    for start in range(0, len(files), 12):
        chunk = files[start:start + 12]
        sheet = Image.new('RGB', (450, 195 * len(chunk)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (aid, path) in enumerate(chunk):
            draw.text((10, i * 195 + 3), aid, fill='black')
            with Image.open(path) as im:
                sheet.paste(im.resize((450, 175)), (0, i * 195 + 20))
        sheet.save(job / 'previews' / f'contact_{start // 12 + 1:02d}.jpg')


def validate_plan(plan, base):
    sources, ids = {}, set()
    for source in plan['sources']:
        sid = identifier(source['id'])
        if sid in sources:
            raise ValueError('Duplicate source ID')
        path = Path(source['path'])
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        sources[sid] = (path, read_image(path).size)
    for a in plan['candidates']:
        a['route'] = {'A': 'AUTO', 'B': 'IMAGE2', 'C': 'MANUAL'}.get(a.get('route'), a.get('route'))
        aid = identifier(a['id'])
        if aid in ids:
            raise ValueError('Duplicate asset ID')
        ids.add(aid)
        if a['route'] not in ('AUTO', 'IMAGE2', 'MANUAL') or not a.get('reason') or not a.get('label'):
            raise ValueError('Missing label/reason or invalid route')
        _, size = sources[a['source_id']]
        box = a['bbox']
        if len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError('bbox needs four integer source coordinates')
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise ValueError('bbox outside image')
        if a['route'] == 'AUTO':
            method = a.get('extraction_method', 'matte')
            if method not in ('matte', 'crop', 'bright-background'):
                raise ValueError('Unknown extraction_method')
            if method == 'matte':
                color(a['background_rgb'])
                tolerance = a.get('background_tolerance', 36)
                if type(tolerance) is not int or not 16 <= tolerance <= 120:
                    raise ValueError('background_tolerance must be an integer in 16..120')
            for p in a.get('background_points', []) + a.get('foreground_points', []):
                if len(p) != 2 or any(type(v) is not int for v in p) or not (x0 <= p[0] < x1 and y0 <= p[1] < y1):
                    raise ValueError('Invalid background point')
        if a['route'] == 'IMAGE2' and not a.get('repair_prompt'):
            raise ValueError('IMAGE2 needs a specific repair_prompt')
        if a.get('repair_mode', 'extract') not in ('extract', 'complete'):
            raise ValueError('repair_mode must be extract or complete')
        if 'repair_allowed' in a and type(a['repair_allowed']) is not bool:
            raise ValueError('repair_allowed must be a boolean')
    if not sources or not ids:
        raise ValueError('Plan must contain sources and candidates')
    return sources


def build(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    sources = validate_plan(plan, plan_path.parent)
    job = Path(args.job).resolve()
    job.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'candidates', 'masks', 'review', 'previews', 'assets', 'review_image2', 'manual', 'rejected', 'repaired']:
        (job / name).mkdir()
    manifest = {'schema_version': 2, 'method': 'codex-discovery-routing-builtin-repair',
                'environment': environment(), 'plan_sha256': digest(plan_path),
                'discovery': plan.get('discovery', {'provider': 'codex-vision', 'coverage': 'not independently measured'}),
                'sources': [], 'assets': []}
    shutil.copyfile(plan_path, job / 'plan.json')
    for sid, (path, size) in sources.items():
        read_image(path).save(job / 'source' / f'{sid}.png')
        manifest['sources'].append({'id': sid, 'original_path': str(path), 'sha256': digest(path), 'size': list(size)})
        manifest['sources'][-1]['snapshot_sha256'] = digest(job / 'source' / f'{sid}.png')
    for candidate in plan['candidates']:
        a = dict(candidate, status='ERROR', generated_repair=bool(candidate.get('generated_source', False)))
        a['initial_route'] = a['route']
        manifest['assets'].append(a)
        try:
            crop = read_image(job / 'source' / f"{a['source_id']}.png").crop(a['bbox'])
            crop.save(job / 'candidates' / f"{a['id']}.png")
            if a['route'] == 'AUTO':
                points = [(x - a['bbox'][0], y - a['bbox'][1]) for x, y in a.get('background_points', [])]
                fg_points = [(x - a['bbox'][0], y - a['bbox'][1]) for x, y in a.get('foreground_points', [])]
                process(job, a, crop, a.get('background_rgb'), points, fg_points)
            elif a['route'] == 'MANUAL' or not a.get('repair_allowed', True):
                manual(job, a, a['reason'] if a['route'] == 'MANUAL' else 'Generative repair disabled')
            else:
                folder = 'review_image2' if a['route'] == 'IMAGE2' else 'manual'
                crop.save(job / folder / f"{a['id']}.png")
                text = f"# {a['label']} ({a['id']})\n\n来源：../source/{a['source_id']}.png\n\n原因：{a['reason']}\n"
                if a['route'] == 'IMAGE2':
                    text += '\n' + repair_prompt(a) + '\n\n默认由 Codex 调用内置图片工具，回图通过 repair-result 导入并验收。\n'
                (job / folder / f"{a['id']}.md").write_text(text, encoding='utf-8')
                a['status'] = 'WAITING_REPAIR' if a['route'] == 'IMAGE2' else 'MANUAL'
        except Exception as exc:
            a.update(status='ERROR', error=str(exc))
            if isinstance(exc, ValueError) and a['route'] == 'AUTO' and a.get('repair_allowed', True) and (job / 'candidates' / f"{a['id']}.png").exists():
                a.update(route='IMAGE2', status='WAITING_REPAIR', repair_mode='extract',
                         routing_history=[{'from': 'AUTO', 'to': 'IMAGE2', 'reason': str(exc)}])
            elif isinstance(exc, ValueError) and a['route'] == 'AUTO' and (job / 'candidates' / f"{a['id']}.png").exists():
                manual(job, a, 'Local extraction failed; generative repair disabled: ' + str(exc))
        candidate_path = job / 'candidates' / f"{a['id']}.png"
        if candidate_path.exists():
            a['candidate_sha256'] = digest(candidate_path)
        save_manifest(job, manifest)
    contact_sheet(job, manifest)
    print(json.dumps({'job': str(job), 'counts': manifest['counts']}, ensure_ascii=False))
    return 1 if manifest['counts']['ERROR'] else 0


def review(args):
    job, manifest = load_job(args.job)
    if not args.note.strip():
        raise ValueError('Review requires an actual observation')
    selected = []
    for aid in dict.fromkeys(args.id):
        identifier(aid)
        a = next(a for a in manifest['assets'] if a['id'] == aid)
        path = job / 'review' / f'{aid}.png'
        if a['status'] != 'REVIEW' or digest(path) != a['review_sha256']:
            raise ValueError('Review stale or asset not awaiting review')
        inspect(read_image(path))
        rejected_name = f"{aid}-attempt-{len(a.get('repair_attempts', []))}.png"
        dest = job / 'assets' / path.name if args.decision == 'accept' else job / 'rejected' / rejected_name
        if dest.exists():
            raise ValueError('Destination already exists')
        selected.append((a, path, dest))
    for a, path, dest in selected:
        path.replace(dest)
        a.update(status='PASS' if args.decision == 'accept' else 'REJECTED',
                 visual_review={'decision': args.decision, 'note': args.note},
                 output_path=str(dest.relative_to(job)), output_sha256=digest(dest))
        if args.decision == 'reject' and a.get('repair_allowed', True) and (a.get('repair_attempts') or a['route'] == 'AUTO'):
            a.update(route='IMAGE2', status='WAITING_REPAIR' if len(a.get('repair_attempts', [])) < 2 else 'MANUAL',
                     repair_feedback=args.note)
        if a.get('repair_attempts'):
            a['repair_attempts'][-1].update(status='PASS' if args.decision == 'accept' else 'REJECTED',
                                           review_note=args.note)
        if args.decision == 'reject' and a['status'] in ('MANUAL', 'REJECTED'):
            manual(job, a, args.note)
        save_manifest(job, manifest)
    print(json.dumps(manifest['counts']))
    return 0


def repaired(args):
    job, manifest = load_job(args.job)
    identifier(args.id)
    a = next(a for a in manifest['assets'] if a['id'] == args.id)
    if a['status'] != 'WAITING_REPAIR':
        raise ValueError('Only WAITING_REPAIR accepts a first repair')
    if not a.get('repair_allowed', True):
        raise ValueError('Generative repair disabled')
    original = Path(args.input).resolve()
    im = read_image(original)
    inspect(matte(im, args.background, args.point))
    dest = job / 'repaired' / f'{a["id"]}.png'
    if dest.exists():
        raise ValueError('Repair exists; preserve it and create a new job')
    im.save(dest)
    process(job, a, im, args.background, args.point)
    a.update(generated_repair=True, repair={'original_path': str(original), 'sha256': digest(original),
             'background_rgb': args.background, 'background_points': args.point})
    save_manifest(job, manifest)
    contact_sheet(job, manifest)
    print(json.dumps(manifest['counts']))
    return 0


def inventory(args):
    files = set()
    for raw in args.input:
        path = Path(raw).resolve()
        if path.is_dir():
            files.update(p.resolve() for p in path.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'))
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError('Input does not exist')
    if not files:
        raise ValueError('No input images')
    sources = [{'id': f'source_{i:03d}', 'path': str(p), 'size': list(read_image(p).size), 'sha256': digest(p)}
               for i, p in enumerate(sorted(files), 1)]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump({'sources': sources, 'discovery': {'provider': 'codex-vision', 'status': 'awaiting_image_analysis'},
                   'candidates': []}, f, ensure_ascii=False, indent=2)
    print(json.dumps({'inventory': str(output), 'sources': sources}, ensure_ascii=False))
    return 0


def repair_prompt(asset):
    request = asset.get('repair_prompt') or f"从参考图中仅提取 {asset['label']}，移除背景和相邻元素。"
    constraint = ('允许补全被遮挡或缺失的部分，保持可见结构、颜色、风格和透视。'
                  if asset.get('repair_mode') == 'complete' else
                  '目标本身完整：只去背景、分离主体，保持可见轮廓、纹理、配色、朝向，不补画或重新设计主体。')
    return (request + '\n' + constraint +
            '\n输出单个可复用组合，四周留出至少 5% 空白边距，使用真正的透明 Alpha 背景。'
            '不要画棋盘格、底板、文字或水印。保留主体内部高光，空隙应透明。' +
            ('\n上次未通过原因：' + asset['repair_feedback'] if asset.get('repair_feedback') else ''))


def repair_queue(args):
    job, manifest = load_job(args.job)
    tasks = []
    for a in manifest['assets']:
        if a['status'] == 'WAITING_REPAIR':
            tasks.append({'id': a['id'], 'label': a['label'], 'provider': 'builtin-image-tool',
                          'model': None, 'attempts_used': len(a.get('repair_attempts', [])),
                          'referenced_image_paths': [str(job / 'candidates' / f"{a['id']}.png")],
                          'prompt': repair_prompt(a)})
    print(json.dumps({'tasks': tasks, 'unresolved': [a['id'] for a in manifest['assets']
                if a['status'] in ('REPAIRING', 'REPAIR_BLOCKED')]}, ensure_ascii=False, indent=2))
    return 0


def repair_start(args):
    job, manifest = load_job(args.job)
    a = next(a for a in manifest['assets'] if a['id'] == args.id)
    attempts = a.setdefault('repair_attempts', [])
    if a['status'] != 'WAITING_REPAIR' or len(attempts) >= 2 or not a.get('repair_allowed', True):
        raise ValueError('Not queued, unresolved request, or two-attempt limit reached')
    candidate = job / 'candidates' / f"{a['id']}.png"
    if a.get('candidate_sha256') and digest(candidate) != a['candidate_sha256']:
        raise ValueError('Candidate changed after build')
    attempts.append({'number': len(attempts) + 1, 'provider': 'builtin-image-tool', 'model': None,
                     'reference_sha256': digest(candidate),
                     'prompt': repair_prompt(a), 'status': 'IN_FLIGHT',
                     'started_at': datetime.now(timezone.utc).isoformat()})
    a['status'] = 'REPAIRING'
    save_manifest(job, manifest)
    print(json.dumps({'id': a['id'], 'attempt': attempts[-1]['number'], 'prompt': attempts[-1]['prompt']}, ensure_ascii=False))
    return 0


def repair_result(args):
    job, manifest = load_job(args.job)
    a = next(a for a in manifest['assets'] if a['id'] == args.id)
    if a['status'] not in ('REPAIRING', 'REPAIR_BLOCKED') or not a.get('repair_attempts'):
        raise ValueError('Start/resume a recorded repair before importing its result')
    attempt = a['repair_attempts'][-1]
    if args.failure:
        # An uncertain provider outcome must not cause a duplicate external request.
        attempt.update(status='BLOCKED', error=args.failure)
        a['status'] = 'REPAIR_BLOCKED'
        save_manifest(job, manifest)
        print(json.dumps({'status': a['status']}))
        return 2
    original = Path(args.input).resolve()
    output = read_image(original)
    dest = job / 'repaired' / f"{a['id']}-attempt-{attempt['number']}.png"
    # Resume a partially imported result only when pixels are identical.
    if dest.exists():
        stored = read_image(dest)
        if stored.size != output.size or not np.array_equal(np.asarray(stored), np.asarray(output)):
            raise ValueError('Attempt artifact differs; preserve history')
    else:
        output.save(dest)
    attempt.update(status='RECEIVED', original_path=str(original), sha256=digest(original),
                   artifact_path=str(dest.relative_to(job)), artifact_sha256=digest(dest),
                   model=args.model, tool_reference=args.tool_reference)
    a.update(generated_repair=True, provider='builtin-image-tool')
    save_manifest(job, manifest)
    try:
        # Preserve native alpha; never re-matte a transparent generated result.
        metrics = inspect(output)
    except ValueError as exc:
        attempt.update(status='QC_FAILED', error=str(exc))
        a.update(status='WAITING_REPAIR' if len(a['repair_attempts']) < 2 else 'MANUAL', repair_feedback=str(exc))
        if a['status'] == 'MANUAL':
            manual(job, a, str(exc))
        save_manifest(job, manifest)
        print(json.dumps({'status': a['status'], 'error': str(exc)}))
        return 1
    aid = a['id']
    output.getchannel('A').save(job / 'masks' / f'{aid}.png')
    output.save(job / 'review' / f'{aid}.png')
    preview(read_image(job / 'candidates' / f'{aid}.png'), output, job / 'previews' / f'{aid}.jpg')
    a.update(status='REVIEW', metrics=metrics, review_sha256=digest(job / 'review' / f'{aid}.png'))
    attempt['status'] = 'REVIEW'
    save_manifest(job, manifest)
    contact_sheet(job, manifest)
    print(json.dumps({'status': 'REVIEW', 'id': aid, 'metrics': metrics}))
    return 0


def status(args):
    job, manifest = load_job(args.job)
    actions = []
    for a in manifest['assets']:
        state = a['status']
        if state == 'REVIEW':
            action = 'view preview, then review accept/reject with observations'
        elif state == 'WAITING_REPAIR':
            action = 'view candidate, repair-start, call image_gen once, repair-result'
        elif state in ('REPAIRING', 'REPAIR_BLOCKED'):
            action = 'recover original tool result; do not dispatch again'
        elif state in ('ERROR', 'REJECTED'):
            action = 'resolve local error; preserve job and report if blocked'
        else:
            continue
        actions.append({'id': a['id'], 'status': state, 'next': action,
                        'candidate': str(job / 'candidates' / f"{a['id']}.png"),
                        'preview': str(job / 'previews' / f"{a['id']}.jpg"),
                        'error': a.get('error')})
    print(json.dumps({'job': str(job), 'counts': manifest['counts'],
                      'ready_to_finalize': not actions, 'actions': actions}, ensure_ascii=False, indent=2))
    return 0


def finalize(args):
    """Only resolved jobs are deliveries. A checksum index allows offline replay."""
    job, manifest = load_job(args.job)
    pending = [a['id'] for a in manifest['assets'] if a['status'] not in ('PASS', 'MANUAL')]
    if pending:
        raise ValueError('Unresolved candidates: ' + ', '.join(pending))
    if manifest.get('plan_sha256') and digest(job / 'plan.json') != manifest['plan_sha256']:
        raise ValueError('Saved plan changed')
    for source in manifest['sources']:
        if source.get('snapshot_sha256') and digest(job / 'source' / f"{source['id']}.png") != source['snapshot_sha256']:
            raise ValueError('Source snapshot changed: ' + source['id'])
    for a in manifest['assets']:
        if a.get('candidate_sha256') and digest(job / 'candidates' / f"{a['id']}.png") != a['candidate_sha256']:
            raise ValueError('Candidate changed: ' + a['id'])
        for attempt in a.get('repair_attempts', []):
            if attempt.get('artifact_path') and digest(job / attempt['artifact_path']) != attempt['artifact_sha256']:
                raise ValueError('Repair artifact changed: ' + a['id'])
        if a['status'] == 'PASS':
            path = job / 'assets' / f"{a['id']}.png"
            if digest(path) != a['output_sha256']:
                raise ValueError('Delivered asset changed: ' + a['id'])
            inspect(read_image(path))
        elif not (job / 'manual' / f"{a['id']}.json").exists():
            raise ValueError('Missing manual handoff: ' + a['id'])
    report = {'environment': manifest.get('environment'), 'discovery': manifest.get('discovery'), 'counts': manifest['counts'],
              'local_pass': sum(a['status'] == 'PASS' and not a['generated_repair'] for a in manifest['assets']),
              'generated_pass': sum(a['status'] == 'PASS' and a['generated_repair'] for a in manifest['assets']),
              'assets': [{'id': a['id'], 'label': a['label'], 'status': a['status'],
                          'generated': a['generated_repair'],
                          'path': 'assets/' + a['id'] + '.png' if a['status'] == 'PASS' else 'manual/' + a['id'] + '.json',
                          'reason': a.get('manual_reason')} for a in manifest['assets']]}
    (job / 'delivery.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    index = {str(p.relative_to(job).as_posix()): digest(p) for p in sorted(job.rglob('*'))
             if p.is_file() and p.name != 'checksums.json'}
    (job / 'checksums.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print(json.dumps({'delivery': str(job / 'delivery.json'), 'files': len(index), 'counts': manifest['counts']}))
    return 0


def verify(args):
    job = Path(args.job).resolve()
    index = json.loads((job / 'checksums.json').read_text(encoding='utf-8'))
    if not index or 'delivery.json' not in index or 'manifest.json' not in index:
        raise ValueError('Incomplete delivery index')
    for relative, expected in index.items():
        path = (job / relative).resolve()
        if not path.is_relative_to(job) or not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed artifact: ' + relative)
    print(json.dumps({'verified_files': len(index), 'job': str(job)}))
    return 0


def self_test():
    # Behavioral check: hole, enclosed pale artwork, review gates, repair, bad inputs.
    with tempfile.TemporaryDirectory(prefix='asset-job-') as tmp:
        root = Path(tmp)
        im = Image.new('RGB', (100, 100), (248, 245, 238))
        d = ImageDraw.Draw(im)
        d.ellipse((15, 15, 85, 85), fill=(20, 80, 180))
        d.ellipse((35, 35, 65, 65), fill=(248, 245, 238))
        d.rectangle((44, 21, 52, 27), fill=(248, 245, 238))
        im.save(root / 'source.png')
        output = matte(im, [248, 245, 238], [(50, 50)])
        assert output.getpixel((0, 0))[3] == 0
        assert output.getpixel((50, 50))[3] == 0
        assert output.getpixel((48, 24))[3] == 255, 'Preserve enclosed pale artwork'
        inspect(output)
        edge_im = Image.new('RGB', (80, 80), (248, 245, 238))
        ed = ImageDraw.Draw(edge_im)
        ed.rectangle((25, 20, 60, 60), fill=(20, 80, 180))
        ed.line((24, 20, 24, 60), fill=(134, 162, 209))
        ed.rectangle((3, 3, 8, 8), fill=(255, 0, 0))
        edge_out = matte(edge_im, [248, 245, 238], foreground_points=[(40, 40)])
        px = edge_out.getpixel((24, 40))
        assert 120 <= px[3] <= 135 and abs(px[0] - 20) <= 3, 'Unmix antialiased backdrop'
        assert edge_out.getpixel((5, 5))[3] == 0, 'Remove unselected neighbor'
        auto = {'id': 'ring', 'source_id': 's', 'label': 'ring', 'bbox': [0, 0, 100, 100],
                'route': 'AUTO', 'reason': 'clear ring', 'background_rgb': [248, 245, 238],
                'background_points': [[50, 50]]}
        pending = dict(auto, id='repair', route='IMAGE2', repair_prompt='Complete ring')
        plan = {'sources': [{'id': 's', 'path': str(root / 'source.png')}],
                'candidates': [auto, pending, dict(auto, id='manual', route='MANUAL')]}
        pp = root / 'plan.json'
        pp.write_text(json.dumps(plan), encoding='utf-8')
        job = root / 'job'
        assert build(argparse.Namespace(plan=pp, job=job)) == 0
        assert not list((job / 'assets').iterdir()), 'No automatic PASS'
        review(argparse.Namespace(job=job, id=['ring'], decision='accept', note='Synthetic ring checked'))
        repaired(argparse.Namespace(job=job, id='repair', input=root / 'source.png', background=[248, 245, 238], point=[[50, 50]]))
        review(argparse.Namespace(job=job, id=['repair'], decision='reject', note='Exercise rejection'))
        _, m = load_job(job)
        assert m['counts']['PASS'] == 1 and m['counts']['MANUAL'] == 2
        assert m['assets'][1]['generated_repair'] is True
        try:
            build(argparse.Namespace(plan=pp, job=job))
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing jobs must not be overwritten')
        plan['candidates'][0]['id'] = '../escape'
        try:
            validate_plan(plan, root)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe IDs must be rejected')
    print('SELF-TEST PASS')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command, function in [('status', status), ('finalize', finalize), ('verify', verify)]:
        p = sub.add_parser(command)
        p.add_argument('--job', required=True)
        p.set_defaults(run=function)
    p = sub.add_parser('build')
    p.add_argument('--plan', required=True)
    p.add_argument('--job', required=True)
    p.set_defaults(run=build)
    p = sub.add_parser('inventory')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(run=inventory)
    p = sub.add_parser('repair-queue')
    p.add_argument('--job', required=True)
    p.set_defaults(run=repair_queue)
    p = sub.add_parser('repair-start')
    p.add_argument('--job', required=True)
    p.add_argument('--id', required=True)
    p.set_defaults(run=repair_start)
    p = sub.add_parser('repair-result')
    p.add_argument('--job', required=True)
    p.add_argument('--id', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input')
    group.add_argument('--failure')
    p.add_argument('--model', default=None, help='Only when explicitly reported by the image tool')
    p.add_argument('--tool-reference', default=None)
    p.set_defaults(run=repair_result)
    p = sub.add_parser('review')
    p.add_argument('--job', required=True)
    p.add_argument('--id', nargs='+', required=True)
    p.add_argument('--decision', choices=['accept', 'reject'], required=True)
    p.add_argument('--note', required=True)
    p.set_defaults(run=review)
    p = sub.add_parser('repaired')
    p.add_argument('--job', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--background', nargs=3, type=int, required=True)
    p.add_argument('--point', nargs=2, type=int, action='append', default=[])
    p.set_defaults(run=repaired)
    sub.add_parser('self-test').set_defaults(run=lambda _: self_test())
    args = parser.parse_args()
    try:
        return args.run(args)
    except Exception as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())

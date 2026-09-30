"""Vision-guided parallel local extraction, optional completion and review."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import shutil
import platform
import math
import errno
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import PIL


_hashes = ContextVar('command_hashes', default=None)


@contextmanager
def hash_scope():
    if _hashes.get() is not None:
        yield
        return
    token = _hashes.set({})
    try:
        yield
    finally:
        _hashes.reset(token)


def hash_command(function):
    @wraps(function)
    def run(args):
        with hash_scope():
            return function(args)
    return run


@contextmanager
def job_lock(path):
    """OS releases this lock even when a CLI process is interrupted."""
    job = Path(path).resolve()
    with (job / '.job.lock').open('a+b') as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b'0')
            handle.flush()
        if platform.system() == 'Windows':
            import msvcrt

            def lock(unlock=False):
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            def lock(unlock=False):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 30
        while True:
            try:
                lock()
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError('Job busy; retry the local command, not the image request') from exc
                time.sleep(0.05)
        try:
            with hash_scope():
                yield
        finally:
            lock(unlock=True)


def locked_job(function):
    @wraps(function)
    def run(args):
        with job_lock(args.job):
            return function(args)
    return run


def repair_capacity(manifest):
    limit = manifest.get('max_parallel', 1)
    active = sum(a['status'] in ('REPAIRING', 'REPAIR_BLOCKED') for a in manifest['assets'])
    return {'max_parallel': limit, 'active': active, 'available_slots': max(0, limit - active)}


def active_repairs(manifest):
    tasks = []
    for a in manifest['assets']:
        if a['status'] in ('REPAIRING', 'REPAIR_BLOCKED'):
            attempt = a.get('repair_attempts', [{}])[-1]
            tasks.append({'id': a['id'], 'status': a['status'], 'attempt': attempt.get('number'),
                          'worker': attempt.get('worker'), 'started_at': attempt.get('started_at'),
                          'tool_reference': attempt.get('tool_reference'), 'error': attempt.get('error')})
    return tasks


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


def file_signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def digest(path, fresh=False):
    path = Path(path).resolve()
    before = file_signature(path)
    memo = _hashes.get()
    if not fresh and memo is not None and memo.get(path, (None,))[0] == before:
        return memo[path][1]
    result = hash_file(path)
    if file_signature(path) != before:
        raise ValueError('File changed while hashing: ' + str(path))
    if memo is not None and not fresh:
        memo[path] = (before, result)
    return result


def read_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGBA')


def image_size(path):
    # Read metadata only. Extraction applies the same EXIF orientation later.
    with Image.open(path) as im:
        return im.size[::-1] if im.getexif().get(274) in (5, 6, 7, 8) else im.size


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


def inspect(im, allow_bottom=False):
    alpha = np.asarray(im.getchannel('A'))
    foreground = alpha > 8
    if not foreground.any() or not (alpha == 0).any():
        raise ValueError('Empty foreground or no fully transparent background')
    if ((alpha[0] > 8).any() or (not allow_bottom and (alpha[-1] > 8).any())
            or (alpha[:, 0] > 8).any() or (alpha[:, -1] > 8).any()):
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
    output = (im.copy() if method == 'native-alpha' else
              matte(im, background, points, foreground_points, asset.get('background_tolerance', 36)) if method == 'matte'
              else local_extract(im, method, asset.get('padding', 12)))
    # Check the actual crop before adding transparent padding. Padding cannot
    # prove that background removal succeeded or that the target is complete.
    inspect(output, allow_bottom=bool(asset.get('portrait') and method == 'native-alpha'))
    if method == 'matte':
        padding = asset.get('padding', 12)
        if type(padding) is not int or not 0 <= padding <= 512:
            raise ValueError('padding must be an integer in 0..512 for matte')
        if padding:
            output = ImageOps.expand(output, border=padding, fill=(0, 0, 0, 0))
    metrics = inspect(output, allow_bottom=bool(asset.get('portrait') and method == 'native-alpha'))
    aid = asset['id']
    output.getchannel('A').save(job / 'masks' / f'{aid}.png')
    output.save(job / 'review' / f'{aid}.png')
    preview(im, output, job / 'previews' / f'{aid}.jpg')
    asset.update(status='REVIEW', metrics=metrics, review_sha256=digest(job / 'review' / f'{aid}.png'),
                 preview_sha256=digest(job / 'previews' / f'{aid}.jpg'))


def compose_portrait(im, layout):
    """Keep native pixels; change canvas and crop only the selected lower torso."""
    if not isinstance(layout, dict):
        raise ValueError('Portrait layout must be a JSON object')
    alpha = np.asarray(im.getchannel('A'))
    inspect(im, allow_bottom=True)
    polygon = layout.get('head_polygon', [])
    if (not isinstance(polygon, list) or len(polygon) < 3
            or any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p)
                   or not (0 <= p[0] < im.width and 0 <= p[1] < im.height) for p in polygon)):
        raise ValueError('head_polygon needs at least three integer points inside the input')
    box = layout.get('upper_body_bbox')
    if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
            or not (0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height)):
        raise ValueError('upper_body_bbox needs four valid input coordinates')
    x0, y0, x1, y1 = box
    if any(not (x0 <= x < x1 and y0 <= y < y1) for x, y in polygon):
        raise ValueError('Upper-body crop must contain the entire head polygon')
    if ((alpha[:y0] > 8).any() or (alpha[:y1, :x0] > 8).any() or (alpha[:y1, x1:] > 8).any()):
        raise ValueError('Upper-body crop may trim the bottom only, not head or shoulders')
    target = layout.get('target_head_area_ratio', 0.4)
    if type(target) not in (int, float) or not math.isfinite(target) or not 0 < target < 1:
        raise ValueError('target_head_area_ratio must be a finite number between zero and one')
    aspect = layout.get('aspect_ratio', [3, 4])
    if (not isinstance(aspect, list) or len(aspect) != 2
            or any(type(v) is not int or not 1 <= v <= 100 for v in aspect)):
        raise ValueError('aspect_ratio needs two positive integers in 1..100')
    divisor = math.gcd(*aspect)
    aw, ah = (v // divisor for v in aspect)
    region = Image.new('L', im.size)
    ImageDraw.Draw(region).polygon([tuple(p) for p in polygon], fill=255)
    head = Image.fromarray(np.where(np.asarray(region) > 0, alpha, 0).astype(np.uint8)).crop(box)
    head_area = float(np.asarray(head, dtype=np.float64).sum() / 255)
    if head_area <= 0:
        raise ValueError('Head polygon has no visible Alpha coverage')
    body = im.crop(box)
    bounds, head_bounds = body.getchannel('A').getbbox(), head.getbbox()
    bx0, by0, bx1, by1 = bounds
    hx0, _, hx1, _ = head_bounds
    center = (hx0 + hx1) / 2
    target_k = max(1, math.ceil(math.sqrt(head_area / (aw * ah * target))))
    k = max(target_k, math.ceil((by1 - by0) / (ah * 0.95)),
            math.ceil(max(center - bx0, bx1 - center) / (aw * 0.47)))
    while True:
        width, height = aw * k, ah * k
        dx, dy = round(width / 2 - center), math.ceil(height * 0.05) - by0
        if (dx + bx0 >= math.ceil(width * 0.03) and width - (dx + bx1) >= math.ceil(width * 0.03)
                and dy + by1 <= height):
            break
        k += 1
    if width > 8192 or height > 8192 or width * height > 64_000_000:
        raise ValueError('Portrait canvas exceeds local memory limit; use a smaller input')
    output = Image.new('RGBA', (width, height))
    output.paste(body, (dx, dy))  # No Alpha mask: preserve RGBA, including soft edges.
    mask = Image.new('L', output.size)
    mask.paste(head, (dx, dy))
    actual = float(np.asarray(mask, dtype=np.float64).sum() / 255 / (width * height))
    metadata = {'target_head_area_ratio': target, 'actual_head_area_ratio': actual,
                'head_area_pixels': head_area, 'area_estimate': 'vision-polygon-intersected-with-alpha',
                'aspect_ratio': [aw, ah], 'canvas_size': [width, height], 'translation': [dx, dy],
                'upper_body_bbox': box, 'within_tolerance': abs(actual - target) <= 0.03,
                'adjustment_reason': 'Canvas enlarged to preserve upper body and margins' if k > target_k else None,
                'allow_bottom_crop': True, 'upscaled': False}
    return output, mask, metadata


@locked_job
def portrait_layout(args):
    job, manifest = load_job(args.job)
    a = next(a for a in manifest['assets'] if a['id'] == args.id)
    if not a.get('portrait') or a['status'] != 'REVIEW' or a.get('pending_review'):
        raise ValueError('Portrait layout requires an unreviewed portrait without pending review')
    config_path = Path(args.layout).resolve()
    config_sha = digest(config_path)
    pending = a.get('pending_portrait')
    if pending and pending['layout_sha256'] != config_sha:
        raise ValueError('Resume the recorded portrait layout before changing parameters')
    previous = pending or a.get('portrait_layout')
    review_path = job / 'review' / f"{a['id']}.png"
    input_path = job / previous['input_path'] if previous else review_path
    input_sha = previous['input_sha256'] if previous else a['review_sha256']
    if digest(input_path) != input_sha:
        raise ValueError('Portrait input changed')
    if not pending and digest(review_path) != a['review_sha256']:
        raise ValueError('Portrait review file changed')
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.get('input_sha256') != input_sha:
        raise ValueError('Layout input_sha256 must match the actual pre-layout portrait')
    source = read_image(input_path)
    output, head, record = compose_portrait(source, config)
    folder = job / 'portraits' / a['id'] / (input_sha[:16] + '-' + config_sha[:16])
    folder.mkdir(parents=True, exist_ok=True)
    saved_input, saved_config = folder / 'input.png', folder / 'layout.json'
    if not saved_input.exists():
        shutil.copyfile(input_path, saved_input)
    if not saved_config.exists():
        shutil.copyfile(config_path, saved_config)
    if digest(saved_input) != input_sha or digest(saved_config) != config_sha:
        raise ValueError('Saved portrait input or layout changed')
    head_path = folder / 'head-mask.png'
    head.save(head_path)
    overlay = Image.new('RGBA', output.size, (255, 60, 60, 0))
    overlay.putalpha(head.point(lambda v: round(v * 0.35)))
    Image.alpha_composite(output, overlay).save(folder / 'head-preview.png')
    record.update(input_path=str(saved_input.relative_to(job)), input_sha256=input_sha,
                  layout_path=str(saved_config.relative_to(job)), layout_sha256=config_sha,
                  head_mask_path=str(head_path.relative_to(job)), head_mask_sha256=digest(head_path),
                  head_preview_path=str((folder / 'head-preview.png').relative_to(job)),
                  generated=bool(a['generated_repair']))
    a['pending_portrait'] = record
    save_manifest(job, manifest)
    output.save(review_path)
    output.getchannel('A').save(job / 'masks' / f"{a['id']}.png")
    preview(read_image(job / 'candidates' / f"{a['id']}.png"), output, job / 'previews' / f"{a['id']}.jpg")
    a.update(portrait_layout=record, metrics=inspect(output, allow_bottom=True), review_sha256=digest(review_path),
             preview_sha256=digest(job / 'previews' / f"{a['id']}.jpg"))
    a.pop('pending_portrait', None)
    contact_sheet(job, manifest)
    save_manifest(job, manifest)
    print(json.dumps({'id': a['id'], 'portrait_layout': record}, ensure_ascii=False))
    return 0


def inspect_asset(job, asset, im):
    portrait = bool(asset.get('portrait'))
    record = asset.get('portrait_layout')
    if portrait:
        if not record or asset.get('pending_portrait'):
            raise ValueError('Run or resume portrait-layout before accepting this portrait')
        for field in ('input', 'layout', 'head_mask'):
            if digest(job / record[field + '_path']) != record[field + '_sha256']:
                raise ValueError('Portrait evidence changed: ' + field)
        with Image.open(job / record['head_mask_path']) as saved_mask:
            head = np.asarray(saved_mask.convert('L'))
            if saved_mask.size != im.size or (head > np.asarray(im.getchannel('A'))).any():
                raise ValueError('Head mask does not match portrait Alpha')
        ratio = float(head.astype(np.float64).sum() / 255 / (im.width * im.height))
        aw, ah = record['aspect_ratio']
        if im.width * ah != im.height * aw or abs(ratio - record['actual_head_area_ratio']) > 1e-10:
            raise ValueError('Portrait area ratio or aspect changed')
    return inspect(im, allow_bottom=portrait and bool(record))


def contact_sheet(job, manifest):
    files = [(a['id'], job / 'previews' / (a['id'] + '.jpg')) for a in manifest['assets']]
    files = [(aid, path) for aid, path in files if path.exists()]
    for start in range(0, len(files), 8):
        chunk = files[start:start + 8]
        sheet = Image.new('RGB', (900, 375 * len(chunk)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (aid, path) in enumerate(chunk):
            draw.text((10, i * 375 + 3), aid, fill='black')
            with Image.open(path) as im:
                sheet.paste(im, (0, i * 375 + 20))
        sheet.save(job / 'previews' / f'contact_{start // 8 + 1:02d}.jpg')
    write_review_sheets(job, manifest)


def review_kind(asset):
    if (asset.get('portrait') or asset.get('generated_repair') or asset.get('requires_detail_review')
            or asset.get('pending_review') or asset.get('pending_portrait')
            or asset.get('route') != 'AUTO' or asset.get('extraction_method', 'matte') not in ('matte', 'crop')
            or not asset.get('preview_sha256')):
        return 'detail'
    bounds = asset.get('metrics', {}).get('foreground_bbox', [0, 0, 0, 0])
    return 'batch' if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) >= 24 else 'detail'


def write_review_sheets(job, manifest, batch_size=8):
    if type(batch_size) is not int or not 1 <= batch_size <= 12:
        raise ValueError('batch-size must be an integer in 1..12')
    assets = [a for a in manifest['assets'] if a['status'] == 'REVIEW' and review_kind(a) == 'batch']
    batches = []
    for start in range(0, len(assets), batch_size):
        chunk = assets[start:start + batch_size]
        rows = []
        for a in chunk:
            for folder, suffix, field in [('candidates', '.png', 'candidate_sha256'),
                                           ('review', '.png', 'review_sha256'), ('previews', '.jpg', 'preview_sha256')]:
                if digest(job / folder / (a['id'] + suffix)) != a[field]:
                    raise ValueError('Batch input changed: ' + a['id'])
            rows.append({k: a[k] for k in ('id', 'candidate_sha256', 'review_sha256', 'preview_sha256')})
        key = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:20]
        image_path, index_path = job / 'previews' / f'fast-{key}.jpg', job / 'previews' / f'fast-{key}.json'
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding='utf-8'))
            if index['rows'] != rows or digest(image_path) != index['image_sha256']:
                raise ValueError('Saved review sheet changed')
        else:
            sheet = Image.new('RGB', (900, 375 * len(rows)), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, a in enumerate(chunk):
                draw.text((10, 375 * i + 3), a['id'] + ' | A / QC checked / batch review', fill='black')
                with Image.open(job / 'previews' / f"{a['id']}.jpg") as im:
                    sheet.paste(im, (0, 375 * i + 20))
            sheet.save(image_path)
            index = {'image': str(image_path.relative_to(job)), 'image_sha256': digest(image_path), 'rows': rows}
            index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')
        batches.append({'index': str(index_path), 'index_sha256': digest(index_path), 'image': str(image_path),
                        'ids': [a['id'] for a in chunk]})
    return batches


@locked_job
def review_sheet(args):
    job, manifest = load_job(args.job)
    batches = write_review_sheets(job, manifest, getattr(args, 'batch_size', 8))
    print(json.dumps({'batches': batches, 'detail_ids': [a['id'] for a in manifest['assets']
                     if a['status'] == 'REVIEW' and review_kind(a) == 'detail']}, ensure_ascii=False))
    return 0


def validate_plan(plan, base):
    sources, ids = {}, set()
    for source in plan['sources']:
        sid = identifier(source['id'])
        if sid in sources:
            raise ValueError('Duplicate source ID')
        path = Path(source['path'])
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        sources[sid] = (path, image_size(path))
    for a in plan['candidates']:
        a['route'] = {'A': 'AUTO', 'B': 'IMAGE2', 'C': 'MANUAL'}.get(a.get('route'), a.get('route'))
        aid = identifier(a['id'])
        if aid in ids:
            raise ValueError('Duplicate asset ID')
        ids.add(aid)
        if a['route'] not in ('AUTO', 'IMAGE2', 'MANUAL') or not a.get('reason') or not a.get('label'):
            raise ValueError('Missing label/reason or invalid route')
        if 'portrait' in a and type(a['portrait']) is not bool:
            raise ValueError('portrait must be a boolean for explicitly requested portrait mode')
        if 'requires_detail_review' in a and type(a['requires_detail_review']) is not bool:
            raise ValueError('requires_detail_review must be a boolean')
        _, size = sources[a['source_id']]
        box = a['bbox']
        if len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError('bbox needs four integer source coordinates')
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise ValueError('bbox outside image')
        if a['route'] == 'AUTO':
            method = a.get('extraction_method', 'matte')
            if method not in ('matte', 'crop', 'bright-background', 'native-alpha'):
                raise ValueError('Unknown extraction_method')
            if method == 'native-alpha' and not a.get('portrait'):
                raise ValueError('native-alpha is reserved for portrait mode')
            padding = a.get('padding', 12)
            if type(padding) is not int or not (0 if method in ('matte', 'native-alpha') else 1) <= padding <= 512:
                raise ValueError('Invalid padding for extraction_method')
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


def build_candidate(job, candidate, source_images=None):
    # Workers own unique asset files; only the build thread writes the manifest.
    a = dict(candidate, status='ERROR', generated_repair=bool(candidate.get('generated_source', False)))
    a.setdefault('initial_route', a['route'])
    try:
        source = (source_images or {}).get(a['source_id'])
        crop = (source if source is not None else read_image(job / 'source' / f"{a['source_id']}.png")).crop(a['bbox'])
        crop.save(job / 'candidates' / f"{a['id']}.png")
        a['candidate_sha256'] = digest(job / 'candidates' / f"{a['id']}.png")
        if a['route'] == 'AUTO':
            points = [(x - a['bbox'][0], y - a['bbox'][1]) for x, y in a.get('background_points', [])]
            fg_points = [(x - a['bbox'][0], y - a['bbox'][1]) for x, y in a.get('foreground_points', [])]
            process(job, a, crop, a.get('background_rgb'), points, fg_points)
        elif a['route'] == 'MANUAL' or not a.get('repair_allowed', True):
            manual(job, a, a['reason'] if a['route'] == 'MANUAL' else 'Generative repair disabled')
        else:
            crop.save(job / 'review_image2' / f"{a['id']}.png")
            text = f"# {a['label']} ({a['id']})\n\n来源：../source/{a['source_id']}.png\n\n原因：{a['reason']}\n"
            text += '\n' + repair_prompt(a) + '\n\n默认由 Codex 调用内置图片工具，回图通过 repair-result 导入并验收。\n'
            (job / 'review_image2' / f"{a['id']}.md").write_text(text, encoding='utf-8')
            a['status'] = 'WAITING_REPAIR'
    except Exception as exc:
        a.update(status='ERROR', error=str(exc))
        if (isinstance(exc, ValueError) and a['route'] == 'AUTO' and a.get('repair_allowed', True)
                and (a.get('portrait') or a.get('extraction_method') != 'bright-background')
                and (job / 'candidates' / f"{a['id']}.png").exists()):
            a.update(route='IMAGE2', status='WAITING_REPAIR',
                     repair_mode=a.get('repair_mode', 'extract') if a.get('portrait') else 'extract',
                     routing_history=[{'from': 'AUTO', 'to': 'IMAGE2', 'reason': str(exc)}])
        elif isinstance(exc, ValueError) and a['route'] == 'AUTO' and (job / 'candidates' / f"{a['id']}.png").exists():
            manual(job, a, 'Local extraction failed; no generative fallback: ' + str(exc))
    return a


@hash_command
def build(args):
    workers = getattr(args, 'workers', 5)
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    processing = getattr(args, 'processing', 'local-first')
    if processing not in ('local-first', 'builtin-repair'):
        raise ValueError('processing must be local-first or builtin-repair')
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    sources = validate_plan(plan, plan_path.parent)
    source_hashes = {sid: digest(path) for sid, (path, _) in sources.items()}
    for source in plan['sources']:
        if source.get('sha256') and source['sha256'] != source_hashes[source['id']]:
            raise ValueError('Discovery source changed; refresh the plan: ' + source['id'])
    for a in plan['candidates']:
        a['initial_route'] = a['route']
        if processing == 'local-first' and not (a.get('portrait') or
                (a['route'] == 'IMAGE2' and a.get('repair_mode') == 'complete')):
            a['repair_allowed'] = False
            if a['route'] == 'IMAGE2':
                reason = 'Local extraction guidance unavailable; generative extraction disabled in local-first'
                a.update(route='MANUAL', reason=a['reason'] + '; ' + reason,
                         routing_history=[{'from': 'IMAGE2', 'to': 'MANUAL', 'reason': reason}])
    job = Path(args.job).resolve()
    job.mkdir(parents=True, exist_ok=False)
    with job_lock(job):
        for name in ['source', 'candidates', 'masks', 'review', 'previews', 'assets', 'review_image2', 'manual', 'rejected', 'repaired']:
            (job / name).mkdir()
        manifest = {'schema_version': 2, 'method': 'codex-discovery-routing-builtin-repair',
                    'max_parallel': workers, 'processing': processing, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'environment': environment(), 'plan_sha256': digest(plan_path),
                    'discovery': plan.get('discovery', {'provider': 'codex-vision', 'coverage': 'not independently measured'}),
                    'sources': [], 'assets': [dict(a, status='ERROR', error='Local processing not completed')
                                             for a in plan['candidates']]}
        shutil.copyfile(plan_path, job / 'plan.json')
        source_images, decoded_bytes = {}, 0
        for sid, (path, size) in sources.items():
            source = read_image(path)
            if source.size != size or digest(path) != source_hashes[sid]:
                raise ValueError('Source changed during build: ' + sid)
            source.save(job / 'source' / f'{sid}.png')
            # ponytail: bounded shared read-only decode cache; very large sources fall back to disk.
            cost = source.width * source.height * 4
            if decoded_bytes + cost <= 256 * 1024 * 1024:
                source_images[sid] = source
                decoded_bytes += cost
            manifest['sources'].append({'id': sid, 'original_path': str(path), 'sha256': source_hashes[sid], 'size': list(size),
                                        'snapshot_sha256': digest(job / 'source' / f'{sid}.png')})
        save_manifest(job, manifest)
        local_workers = min(workers, len(plan['candidates']))
        with ThreadPoolExecutor(max_workers=local_workers) as pool:
            futures = {pool.submit(build_candidate, job, a, source_images): i for i, a in enumerate(plan['candidates'])}
            for future in as_completed(futures):
                manifest['assets'][futures[future]] = future.result()
                save_manifest(job, manifest)
        contact_sheet(job, manifest)
        print(json.dumps({'job': str(job), 'counts': manifest['counts'], 'local_workers': local_workers,
                          'decoded_source_cache_count': len(source_images),
                          'capacity': repair_capacity(manifest)}, ensure_ascii=False))
        return 1 if manifest['counts']['ERROR'] else 0


def prepare_review(job, asset, decision, note):
    if decision not in ('accept', 'reject') or not isinstance(note, str) or not note.strip():
        raise ValueError('Review requires an actual observation')
    aid = identifier(asset['id'])
    if asset.get('pending_portrait'):
        raise ValueError('Resume portrait-layout before reviewing')
    path = job / 'review' / f'{aid}.png'
    rejected_name = f"{aid}-attempt-{len(asset.get('repair_attempts', []))}.png"
    dest = job / 'assets' / path.name if decision == 'accept' else job / 'rejected' / rejected_name
    pending = {'decision': decision, 'note': note, 'path': str(dest.relative_to(job))}
    if asset['status'] != 'REVIEW' or (asset.get('pending_review') and asset['pending_review'] != pending):
        raise ValueError('Resume the recorded review decision and note before changing it')
    checked = path if path.exists() else dest
    if not path.exists() and not asset.get('pending_review'):
        raise ValueError('Missing review file without a recorded decision')
    if digest(checked) != asset['review_sha256']:
        raise ValueError('Review stale or asset not awaiting review')
    if decision == 'accept':
        inspect_asset(job, asset, read_image(checked))
    else:
        inspect(read_image(checked), allow_bottom=bool(asset.get('portrait')))
    if path.exists() and dest.exists():
        raise ValueError('Destination already exists')
    return asset, path, dest, pending


def commit_reviews(job, manifest, selected):
    # Record the whole transaction before moving any files. Resume each pending
    # decision with its original note if a batch is interrupted midway.
    for a, _, _, pending in selected:
        a['pending_review'] = pending
    save_manifest(job, manifest)
    for a, path, dest, pending in selected:
        decision, note = pending['decision'], pending['note']
        if path.exists():
            path.replace(dest)
        a.update(status='PASS' if decision == 'accept' else 'REJECTED',
                 visual_review={'decision': decision, 'note': note, 'batch': a.get('batch_review'),
                                'mode': 'batch' if a.get('batch_review') and not a.get('requires_detail_review') else 'detail'},
                 output_path=str(dest.relative_to(job)), output_sha256=a['review_sha256'])
        if (decision == 'reject' and a.get('repair_allowed', True)
                and (a.get('portrait') or a.get('extraction_method') != 'bright-background')
                and (a.get('repair_attempts') or a['route'] == 'AUTO')):
            a.update(route='IMAGE2', status='WAITING_REPAIR' if len(a.get('repair_attempts', [])) < 2 else 'MANUAL',
                     repair_feedback=note)
        if a.get('repair_attempts'):
            a['repair_attempts'][-1].update(status='PASS' if decision == 'accept' else 'REJECTED', review_note=note)
        a.pop('pending_review', None)
        if decision == 'reject' and a['status'] in ('MANUAL', 'REJECTED'):
            manual(job, a, note)
    save_manifest(job, manifest)


@locked_job
def review(args):
    job, manifest = load_job(args.job)
    selected = [prepare_review(job, next(a for a in manifest['assets'] if a['id'] == aid), args.decision, args.note)
                for aid in dict.fromkeys(args.id)]
    commit_reviews(job, manifest, selected)
    print(json.dumps(manifest['counts']))
    return 0


@locked_job
def review_batch(args):
    job, manifest = load_job(args.job)
    index_path = Path(args.sheet).resolve()
    if not index_path.is_relative_to(job):
        raise ValueError('Batch index must belong to this job')
    data = json.loads(Path(args.decisions).read_text(encoding='utf-8-sig'))
    index_sha = digest(index_path)
    if data.get('index_sha256') != index_sha:
        raise ValueError('Batch decisions refer to a different index')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    image_path = (job / index['image']).resolve()
    if not image_path.is_relative_to(job) or digest(image_path) != index['image_sha256']:
        raise ValueError('Batch image changed')
    decisions = data['decisions']
    if (len(decisions) != len(index['rows']) or len({d['id'] for d in decisions}) != len(decisions)
            or {d['id'] for d in decisions} != {r['id'] for r in index['rows']}):
        raise ValueError('Batch decisions must cover every indexed ID exactly once')
    by_id = {d['id']: d for d in decisions}
    info = {'index': str(index_path.relative_to(job)), 'index_sha256': index_sha, 'image_sha256': index['image_sha256']}
    selected, details = [], []
    for row in index['rows']:
        a = next(a for a in manifest['assets'] if a['id'] == row['id'])
        d = by_id[a['id']]
        decision = {'PASS': 'accept', 'REJECT': 'reject', 'DETAIL': 'detail'}.get(d['decision'], d['decision'])
        note = d.get('note')
        if decision not in ('accept', 'reject', 'detail') or not isinstance(note, str) or not note.strip():
            raise ValueError('Each batch decision needs a valid decision and actual observation')
        if any(a.get(k) != row[k] for k in ('candidate_sha256', 'review_sha256', 'preview_sha256')):
            raise ValueError('Batch review is stale: ' + a['id'])
        if digest(job / 'candidates' / f"{a['id']}.png") != row['candidate_sha256']:
            raise ValueError('Batch source changed')
        if digest(job / 'previews' / f"{a['id']}.jpg") != row['preview_sha256']:
            raise ValueError('Batch preview changed')
        if a['status'] != 'REVIEW':
            previous = a.get('visual_review', {})
            if a.get('batch_review') != info or previous.get('decision') != decision or previous.get('note') != note:
                raise ValueError('Cannot change a completed batch decision')
            if digest(job / a['output_path']) != a['output_sha256']:
                raise ValueError('Completed batch output changed')
            continue
        if review_kind(a) != 'batch' and a.get('batch_review') != info:
            raise ValueError('This asset requires individual review')
        if decision == 'detail':
            if a.get('pending_review') or digest(job / 'review' / f"{a['id']}.png") != row['review_sha256']:
                raise ValueError('Resume pending review before requesting detail')
            details.append((a, note))
        else:
            selected.append(prepare_review(job, a, decision, note))
    for a, _, _, _ in selected:
        a['batch_review'] = info
    for a, note in details:
        a.update(requires_detail_review=True, detail_note=note, batch_review=info)
    commit_reviews(job, manifest, selected)
    print(json.dumps({'counts': manifest['counts'], 'detail_ids': [a['id'] for a, _ in details]}))
    return 0


@locked_job
def repaired(args):
    job, manifest = load_job(args.job)
    if 'max_parallel' in manifest:
        raise ValueError('New jobs require repair-start and repair-result --attempt; repaired is legacy-only')
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


DISCOVERY_VERSION = 1


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        temporary = Path(stream.name)
    temporary.replace(path)


@hash_command
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
    proxy_size = getattr(args, 'proxy_size', 1024)
    if type(proxy_size) is not int or not 256 <= proxy_size <= 2048:
        raise ValueError('proxy-size must be an integer in 256..2048')
    intent = getattr(args, 'intent', 'reusable-design-assets')
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError('Discovery intent must describe the actual request')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Inventory output already exists')
    sources = [{'id': f'source_{i:03d}', 'path': str(p), 'size': list(image_size(p)), 'sha256': digest(p)}
               for i, p in enumerate(sorted(files), 1)]
    profile = {'version': DISCOVERY_VERSION, 'script_sha256': digest(__file__), 'intent': intent,
               'proxy_size': proxy_size, 'sources': [{k: s[k] for k in ('id', 'sha256', 'size')} for s in sources]}
    key = json_digest(profile)
    cache_root = Path(getattr(args, 'cache_dir', None) or Path.home() / '.cache' / 'design-asset-extractor' / 'discovery').resolve()
    entry = cache_root / key
    entry.mkdir(parents=True, exist_ok=True)
    with job_lock(entry):
        proxies_path = entry / 'proxies.json'
        try:
            proxies = json.loads(proxies_path.read_text(encoding='utf-8'))
            if not isinstance(proxies, dict):
                proxies = {}
        except (OSError, ValueError):
            proxies = {}
        for source in sources:
            path = entry / f"{source['id']}.png"
            previous = proxies.get(source['id'], {})
            if (not isinstance(previous, dict) or not isinstance(previous.get('size'), list)
                    or len(previous['size']) != 2 or any(type(v) is not int or v <= 0 for v in previous['size'])):
                previous = {}
            if (not path.exists() or digest(path) != previous.get('sha256')
                    or list(image_size(path)) != previous.get('size')):
                im = read_image(source['path'])
                if im.size != tuple(source['size']) or digest(source['path']) != source['sha256']:
                    raise ValueError('Source changed during discovery')
                im.thumbnail((proxy_size, proxy_size), Image.Resampling.LANCZOS)
                im.save(path)
                previous = {'sha256': digest(path), 'size': list(im.size)}
                proxies[source['id']] = previous
            source.update(proxy_path=str(path), proxy_sha256=previous['sha256'], proxy_size=previous['size'],
                          proxy_to_source=[source['size'][i] / previous['size'][i] for i in (0, 1)])
        atomic_json(proxies_path, proxies)
        plan = None
        if not getattr(args, 'refresh_discovery', False):
            try:
                saved = json.loads((entry / 'plan.json').read_text(encoding='utf-8'))
                if saved['profile'] == profile and json_digest(saved['plan']) == saved['plan_sha256']:
                    plan = saved['plan']
                    plan['sources'] = sources
                    validate_plan(plan, output.parent)
            except (OSError, ValueError, KeyError, TypeError):
                plan = None
        if plan is None:
            plan = {'sources': sources, 'discovery': {'provider': 'codex-vision', 'status': 'awaiting_image_analysis'},
                    'candidates': []}
        plan.setdefault('discovery', {}).update(cache_hit=bool(plan['candidates']), intent=intent)
        plan['discovery_cache'] = {'directory': str(entry), 'key': key, 'profile': profile}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(json.dumps({'inventory': str(output), 'sources': sources, 'cache_hit': plan['discovery']['cache_hit'],
                      'candidates': len(plan['candidates'])}, ensure_ascii=False))
    return 0


@hash_command
def discovery_save(args):
    inventory_path, plan_path = Path(args.inventory).resolve(), Path(args.plan).resolve()
    inventory_data = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    cache = inventory_data['discovery_cache']
    if json_digest(cache['profile']) != cache['key'] or cache['profile']['script_sha256'] != digest(__file__):
        raise ValueError('Discovery cache profile changed; regenerate inventory')
    current = {s['id']: s for s in inventory_data['sources']}
    if ([{k: s[k] for k in ('id', 'sha256', 'size')} for s in current.values()] != cache['profile']['sources']
            or cache['profile']['version'] != DISCOVERY_VERSION):
        raise ValueError('Inventory no longer matches its discovery profile')
    if {s['id'] for s in plan['sources']} != set(current):
        raise ValueError('Discovery plan must use the inventoried sources')
    for s in plan['sources']:
        path = Path(s['path'])
        path = (plan_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if path != Path(current[s['id']]['path']).resolve():
            raise ValueError('Discovery plan refers to a different source path')
    for s in current.values():
        if digest(s['path']) != s['sha256'] or list(image_size(s['path'])) != s['size']:
            raise ValueError('Discovery source changed')
        if digest(s['proxy_path']) != s['proxy_sha256'] or list(image_size(s['proxy_path'])) != s['proxy_size']:
            raise ValueError('Discovery proxy changed')
    space = plan.pop('coordinate_space', 'source')
    if space not in ('source', 'proxy'):
        raise ValueError('coordinate_space must be source or proxy')
    if space == 'proxy':
        for a in plan['candidates']:
            s = current[a['source_id']]
            pw, ph = s['proxy_size']
            box = a['bbox']
            if (len(box) != 4 or any(type(v) is not int for v in box)
                    or not (0 <= box[0] < box[2] <= pw and 0 <= box[1] < box[3] <= ph)):
                raise ValueError('Proxy bbox outside discovery thumbnail')
            sx, sy = (s['size'][i] / s['proxy_size'][i] for i in (0, 1))
            a['bbox'] = [math.floor(box[0] * sx), math.floor(box[1] * sy),
                         min(s['size'][0], math.ceil(box[2] * sx)), min(s['size'][1], math.ceil(box[3] * sy))]
            for field in ('foreground_points', 'background_points'):
                mapped = []
                for x, y in a.get(field, []):
                    if type(x) is not int or type(y) is not int or not (0 <= x < pw and 0 <= y < ph):
                        raise ValueError('Proxy point outside thumbnail')
                    mapped.append([min(s['size'][0] - 1, round(x * sx)), min(s['size'][1] - 1, round(y * sy))])
                if field in a:
                    a[field] = mapped
    plan['sources'] = list(current.values())
    if plan.get('discovery', {}).get('status') != 'complete':
        raise ValueError('Only completed visual discovery may be cached')
    validate_plan(plan, plan_path.parent)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Normalized plan already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    entry = Path(cache['directory']).resolve()
    with job_lock(entry):
        atomic_json(entry / 'plan.json', {'profile': cache['profile'], 'plan': plan, 'plan_sha256': json_digest(plan)})
    print(json.dumps({'plan': str(output), 'cache_saved': True, 'coordinate_space': 'source'}, ensure_ascii=False))
    return 0


def repair_prompt(asset):
    request = asset.get('repair_prompt') or f"从参考图中仅提取 {asset['label']}，移除背景和相邻元素。"
    constraint = ('允许补全被遮挡或缺失的部分，保持可见结构、颜色、风格和透视。'
                  if asset.get('repair_mode') == 'complete' else
                  '目标本身完整：只去背景、分离主体，保持可见轮廓、纹理、配色、朝向，不补画或重新设计主体。')
    portrait = ('\n人像模式：保持本人身份、五官、发型、表情、原姿态及可见服装，'
                '不美颜、不换装、不将侧脸改成正脸。仅补明确缺失的肩部和胸部上段。'
                '完整保留头发、双肩和上半身，输出真实透明人像。'
                '60%仅为可选参考，不需生成中间版；最终头部面积约40%由后续脚本构图。'
                if asset.get('portrait') else '')
    return (request + '\n' + constraint + portrait +
            '\n输出单个可复用组合，四周留出至少 5% 空白边距，使用真正的透明 Alpha 背景。'
            '不要画棋盘格、底板、文字或水印。保留主体内部高光，空隙应透明。' +
            ('\n上次未通过原因：' + asset['repair_feedback'] if asset.get('repair_feedback') else ''))


@locked_job
def repair_queue(args):
    job, manifest = load_job(args.job)
    tasks = []
    for a in manifest['assets']:
        if a['status'] == 'WAITING_REPAIR':
            tasks.append({'id': a['id'], 'label': a['label'], 'provider': 'builtin-image-tool',
                          'model': None, 'attempts_used': len(a.get('repair_attempts', [])),
                          'referenced_image_paths': [str(job / 'candidates' / f"{a['id']}.png")],
                          'prompt': repair_prompt(a)})
    print(json.dumps({'tasks': tasks, 'processing': manifest.get('processing', 'builtin-repair'),
                'capacity': repair_capacity(manifest), 'active_tasks': active_repairs(manifest),
                'unresolved': [a['id'] for a in manifest['assets']
                if a['status'] in ('REPAIRING', 'REPAIR_BLOCKED')]}, ensure_ascii=False, indent=2))
    return 0


@locked_job
def repair_start(args):
    job, manifest = load_job(args.job)
    a = next(a for a in manifest['assets'] if a['id'] == args.id)
    attempts = a.setdefault('repair_attempts', [])
    if a['status'] != 'WAITING_REPAIR' or len(attempts) >= 2 or not a.get('repair_allowed', True):
        raise ValueError('Not queued, unresolved request, or two-attempt limit reached')
    if not repair_capacity(manifest)['available_slots']:
        raise ValueError('Parallel capacity full; finish or recover existing requests first')
    candidate = job / 'candidates' / f"{a['id']}.png"
    if a.get('candidate_sha256') and digest(candidate) != a['candidate_sha256']:
        raise ValueError('Candidate changed after build')
    attempts.append({'number': len(attempts) + 1, 'provider': 'builtin-image-tool', 'model': None,
                     'worker': getattr(args, 'worker', 'main'),
                     'reference_sha256': digest(candidate),
                     'prompt': repair_prompt(a), 'status': 'IN_FLIGHT',
                     'started_at': datetime.now(timezone.utc).isoformat()})
    a['status'] = 'REPAIRING'
    save_manifest(job, manifest)
    print(json.dumps({'id': a['id'], 'attempt': attempts[-1]['number'], 'prompt': attempts[-1]['prompt'],
                      'referenced_image_paths': [str(candidate)]}, ensure_ascii=False))
    return 0


@locked_job
def repair_result(args):
    job, manifest = load_job(args.job)
    a = next(a for a in manifest['assets'] if a['id'] == args.id)
    if a['status'] not in ('REPAIRING', 'REPAIR_BLOCKED') or not a.get('repair_attempts'):
        raise ValueError('Start/resume a recorded repair before importing its result')
    attempt = a['repair_attempts'][-1]
    number = getattr(args, 'attempt', None)
    if ('max_parallel' in manifest and number is None) or (number is not None and number != attempt['number']):
        raise ValueError('Missing or stale attempt number; use the original repair-start result')
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
    attempt.setdefault('received_at', datetime.now(timezone.utc).isoformat())
    a.update(generated_repair=True, provider='builtin-image-tool')
    a.pop('portrait_layout', None)
    a.pop('batch_review', None)
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
    a['preview_sha256'] = digest(job / 'previews' / f'{aid}.jpg')
    attempt['status'] = 'REVIEW'
    save_manifest(job, manifest)
    contact_sheet(job, manifest)
    print(json.dumps({'status': 'REVIEW', 'id': aid, 'metrics': metrics}))
    return 0


@locked_job
def status(args):
    job, manifest = load_job(args.job)
    actions = []
    for a in manifest['assets']:
        state = a['status']
        if state == 'REVIEW':
            action = ('resume review with recorded decision and note' if a.get('pending_review') else
                      'resume portrait-layout with recorded layout parameters' if a.get('pending_portrait') else
                      'view actual result, mark head polygon and upper-body crop, then portrait-layout' if
                      a.get('portrait') and not a.get('portrait_layout') else
                      'view Contact Sheet and review-batch; send uncertain items to detail' if review_kind(a) == 'batch' else
                      'view original and individual preview, then review accept/reject with observations')
        elif state == 'WAITING_REPAIR':
            action = ('view candidate, repair-start to reserve a slot, call image_gen once, repair-result with attempt'
                      if repair_capacity(manifest)['available_slots'] else 'wait for capacity; recover blocked requests')
        elif state in ('REPAIRING', 'REPAIR_BLOCKED'):
            action = 'recover original tool result; do not dispatch again'
        elif state in ('ERROR', 'REJECTED'):
            action = 'resolve local error; preserve job and report if blocked'
        else:
            continue
        actions.append({'id': a['id'], 'status': state, 'next': action,
                        'candidate': str(job / 'candidates' / f"{a['id']}.png"),
                         'preview': str(job / 'previews' / f"{a['id']}.jpg"),
                        'pending_review': a.get('pending_review'),
                        'pending_portrait': a.get('pending_portrait'), 'portrait_layout': a.get('portrait_layout'),
                        'error': a.get('error')})
    print(json.dumps({'job': str(job), 'counts': manifest['counts'],
                      'processing': manifest.get('processing', 'builtin-repair'),
                      'batch_review_ids': [a['id'] for a in manifest['assets']
                                           if a['status'] == 'REVIEW' and review_kind(a) == 'batch'],
                      'review_sheet_command': 'review-sheet --job ' + str(job),
                      'capacity': repair_capacity(manifest), 'active_tasks': active_repairs(manifest),
                      'ready_to_finalize': not actions, 'actions': actions}, ensure_ascii=False, indent=2))
    return 0


@locked_job
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
            inspect_asset(job, a, read_image(path))
        else:
            for suffix in ('png', 'md', 'json'):
                if not (job / 'manual' / f"{a['id']}.{suffix}").is_file():
                    raise ValueError('Missing manual handoff: ' + a['id'] + '.' + suffix)
            if digest(job / 'manual' / f"{a['id']}.png") != digest(job / 'candidates' / f"{a['id']}.png"):
                raise ValueError('Manual crop changed: ' + a['id'])
            handoff = json.loads((job / 'manual' / f"{a['id']}.json").read_text(encoding='utf-8'))
            if handoff.get('id') != a['id'] or handoff.get('status') != 'MANUAL':
                raise ValueError('Invalid manual handoff: ' + a['id'])
    report = {'environment': manifest.get('environment'), 'discovery': manifest.get('discovery'), 'counts': manifest['counts'],
              'processing': manifest.get('processing', 'builtin-repair'),
              'max_parallel': manifest.get('max_parallel', 1),
              'local_pass': sum(a['status'] == 'PASS' and not a['generated_repair'] for a in manifest['assets']),
              'generated_pass': sum(a['status'] == 'PASS' and a['generated_repair'] for a in manifest['assets']),
              'assets': [{'id': a['id'], 'label': a['label'], 'status': a['status'],
                          'generated': a['generated_repair'],
                          'review_mode': a.get('visual_review', {}).get('mode'), 'batch_review': a.get('batch_review'),
                          'portrait_layout': a.get('portrait_layout'),
                          'path': 'assets/' + a['id'] + '.png' if a['status'] == 'PASS' else 'manual/' + a['id'] + '.json',
                          'reason': a.get('manual_reason')} for a in manifest['assets']]}
    (job / 'delivery.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    index = {str(p.relative_to(job).as_posix()): digest(p) for p in sorted(job.rglob('*'))
             if p.is_file() and p.name != 'checksums.json' and p != job / '.job.lock'}
    (job / 'checksums.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print(json.dumps({'delivery': str(job / 'delivery.json'), 'files': len(index), 'counts': manifest['counts']}))
    return 0


@locked_job
def verify(args):
    job = Path(args.job).resolve()
    index = json.loads((job / 'checksums.json').read_text(encoding='utf-8'))
    if not index or 'delivery.json' not in index or 'manifest.json' not in index:
        raise ValueError('Incomplete delivery index')
    for relative, expected in index.items():
        path = (job / relative).resolve()
        if not path.is_relative_to(job) or not path.is_file() or digest(path, fresh=True) != expected:
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
        assert build(argparse.Namespace(plan=pp, job=job, processing='builtin-repair')) == 0
        assert not list((job / 'assets').iterdir()), 'No automatic PASS'
        review(argparse.Namespace(job=job, id=['ring'], decision='accept', note='Synthetic ring checked'))
        # Exercise the historical hand-import path using an old-format job.
        with job_lock(job):
            _, legacy = load_job(job)
            legacy.pop('max_parallel')
            save_manifest(job, legacy)
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
    p.add_argument('--workers', type=int, default=5, help='Positive concurrency limit; default 5')
    p.add_argument('--processing', choices=['local-first', 'builtin-repair'], default='local-first',
                   help='Local scripts by default; explicit completion or portrait mode may use image tools')
    p.set_defaults(run=build)
    p = sub.add_parser('inventory')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--proxy-size', type=int, default=1024, help='Discovery thumbnail long edge (256..2048)')
    p.add_argument('--intent', default='reusable-design-assets', help='Actual request; change when goals or constraints change')
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--refresh-discovery', action='store_true')
    p.set_defaults(run=inventory)
    p = sub.add_parser('discovery-save')
    p.add_argument('--inventory', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(run=discovery_save)
    p = sub.add_parser('review-sheet')
    p.add_argument('--job', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.set_defaults(run=review_sheet)
    p = sub.add_parser('review-batch')
    p.add_argument('--job', required=True)
    p.add_argument('--sheet', required=True)
    p.add_argument('--decisions', required=True)
    p.set_defaults(run=review_batch)
    p = sub.add_parser('repair-queue')
    p.add_argument('--job', required=True)
    p.set_defaults(run=repair_queue)
    p = sub.add_parser('repair-start')
    p.add_argument('--job', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--worker', default='main', help='Worker/agent label for tracing the request')
    p.set_defaults(run=repair_start)
    p = sub.add_parser('repair-result')
    p.add_argument('--job', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--attempt', type=int, help='Required for new jobs; number returned by repair-start')
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
    p = sub.add_parser('portrait-layout')
    p.add_argument('--job', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--layout', required=True, help='Vision-provided head polygon and upper-body crop JSON')
    p.set_defaults(run=portrait_layout)
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

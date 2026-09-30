"""Offline portrait geometry and workflow checks; no real face or provider call."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import asset_job as j


def call(fn, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        code = fn(argparse.Namespace(**kwargs))
    return code, json.loads(stream.getvalue()) if stream.getvalue().strip() else None


def invalid(fn, **kwargs):
    job = kwargs.get('job')
    before = j.digest(Path(job) / 'manifest.json') if job else None
    try:
        call(fn, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid portrait operation accepted')
    if before:
        assert before == j.digest(Path(job) / 'manifest.json')


def fixture(height=260):
    im = Image.new('RGBA', (300, height))
    d = ImageDraw.Draw(im)
    d.rectangle((70, 20, 229, 179), fill=(150, 110, 90, 255))
    d.polygon([(130, 180), (169, 180), (224, 210), (224, height - 1),
               (75, height - 1), (75, 210)], fill=(35, 70, 100, 255))
    im.putpixel((70, 40), (150, 110, 90, 128))
    return im


def config(height=260):
    return {'head_polygon': [[69, 19], [230, 19], [230, 180], [69, 180]],
            'upper_body_bbox': [0, 0, 300, height]}


def geometry_checks():
    source = fixture()
    output, mask, record = j.compose_portrait(source, config())
    assert output.width * 4 == output.height * 3
    assert record['within_tolerance'] and not record['upscaled']
    expected = float(np.asarray(source.getchannel('A'))[19:181, 69:231].astype(np.float64).sum() / 255)
    assert abs(record['head_area_pixels'] - expected) < 1e-10
    assert record['head_area_pixels'] != 162 * 162, 'Transparent and soft pixels counted as opaque'
    assert abs(record['actual_head_area_ratio'] - expected / (output.width * output.height)) < 1e-10
    assert abs(record['actual_head_area_ratio'] - 160 / output.height) > .05, 'Used head height instead of area'
    dx, dy = record['translation']
    original = np.asarray(source)
    oy, ox = np.where(original[:, :, 3] > 0)
    assert np.array_equal(original[oy, ox], np.asarray(output)[oy + dy, ox + dx]), 'Portrait pixels stretched or changed'
    assert (np.asarray(mask) <= np.asarray(output.getchannel('A'))).all()
    wide = source.copy()
    ImageDraw.Draw(wide).rectangle((10, 215, 289, 259), fill=(35, 70, 100, 255))
    _, _, wider = j.compose_portrait(wide, config())
    assert wider['adjustment_reason'] and wider['actual_head_area_ratio'] < .37
    tall = fixture(420)
    cropped, _, lower = j.compose_portrait(tall, config(400))
    assert lower['adjustment_reason'] and lower['actual_head_area_ratio'] < .37
    assert np.asarray(cropped.getchannel('A'))[-1].max() > 8, 'Missing intentional chest crop'
    try:
        j.inspect(cropped)
    except ValueError:
        pass
    else:
        raise AssertionError('Normal asset accepted bottom-edge foreground')
    j.inspect(cropped, allow_bottom=True)
    for bad in [dict(config(), head_polygon=[]), dict(config(), head_polygon=[[0, 0], [300, 0], [1, 1]]),
                dict(config(), upper_body_bbox=[80, 0, 300, 260]),
                dict(config(), target_head_area_ratio=True), dict(config(), target_head_area_ratio=float('nan')),
                dict(config(), target_head_area_ratio=0), dict(config(), aspect_ratio=[0, 4])]:
        try:
            j.compose_portrait(source, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid geometry accepted')
    return {'head_area_pixels': expected, 'actual_head_area_ratio': record['actual_head_area_ratio'],
            'wide_ratio': wider['actual_head_area_ratio'], 'tall_ratio': lower['actual_head_area_ratio'],
            'pixels_preserved': True, 'bottom_crop_scoped': True}


def workflow_checks(root):
    source = fixture(420)
    source.save(root / 'source.png')
    base = {'source_id': 's', 'label': 'Synthetic portrait silhouette, not a real face',
            'bbox': [0, 0, 300, 420], 'route': 'A', 'reason': 'Offline fixture only',
            'portrait': True, 'extraction_method': 'native-alpha', 'padding': 0}
    plan = {'sources': [{'id': 's', 'path': str(root / 'source.png')}],
            'discovery': {'provider': 'synthetic-offline-test', 'note': 'No real portrait; no provider call'},
            'candidates': [dict(base, id='native'), dict(base, id='native_other'),
                           dict(base, id='extract', route='B', repair_prompt='Isolate synthetic subject'),
                           dict(base, id='complete', route='B', repair_mode='complete', repair_prompt='Complete synthetic torso'),
                           dict(base, id='ordinary', route='B', portrait=False, repair_prompt='Ordinary extraction'),
                           dict(base, id='strict', route='B', repair_allowed=False, repair_prompt='Strict subject')]}
    plan_path = root / 'plan.json'
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    job = root / 'job'
    call(j.build, plan=plan_path, job=job, workers=5)
    m = j.load_job(job)[1]
    assert m['counts']['REVIEW'] == 2 and m['counts']['WAITING_REPAIR'] == 2 and m['counts']['MANUAL'] == 2
    assert not any(a.get('repair_attempts') for a in m['assets'])
    invalid(j.review, job=job, id=['native'], decision='accept', note='Missing required layout')
    _, status = call(j.status, job=job)
    assert 'portrait-layout' in next(a for a in status['actions'] if a['id'] == 'native')['next']
    native_input = job / 'review' / 'native.png'
    layout = dict(config(400), input_sha256=j.digest(native_input))
    layout_path = root / 'layout.json'
    layout_path.write_text(json.dumps(layout), encoding='utf-8')
    bad_path = root / 'stale-layout.json'
    bad_path.write_text(json.dumps(dict(layout, input_sha256='wrong')), encoding='utf-8')
    invalid(j.portrait_layout, job=job, id='native', layout=bad_path)
    invalid(j.portrait_layout, job=job, id='ordinary', layout=layout_path)
    with patch.object(j, 'preview', side_effect=OSError('Injected layout preview interruption')):
        try:
            call(j.portrait_layout, job=job, id='native', layout=layout_path)
        except OSError:
            pass
        else:
            raise AssertionError('Interruption not reached')
    assert j.load_job(job)[1]['assets'][0]['pending_portrait']
    invalid(j.review, job=job, id=['native'], decision='reject', note='Cannot bypass pending layout')
    invalid(j.portrait_layout, job=job, id='native', layout=bad_path)
    call(j.portrait_layout, job=job, id='native', layout=layout_path)
    first_sha = j.digest(native_input)
    call(j.portrait_layout, job=job, id='native', layout=layout_path)
    assert j.digest(native_input) == first_sha, 'Resuming accumulated crop or Alpha changes'
    # Exercise the actual CLI and independent workers on different portraits.
    argv = [sys.executable, str(Path(j.__file__)), 'portrait-layout', '--job', str(job), '--layout', str(layout_path)]
    workers = [subprocess.Popen([*argv, '--id', aid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
               for aid in ('native', 'native_other')]
    for worker in workers:
        out, err = worker.communicate(timeout=30)
        assert worker.returncode == 0, err.decode('utf-8')
    for aid in ('extract', 'complete'):
        _, started = call(j.repair_start, job=job, id=aid)
        assert '本人身份' in started['prompt'] and '40%' in started['prompt']
        native = root / f'{aid}-saved-fixture.png'
        ImageOps.expand(source, 4, fill=(0, 0, 0, 0)).save(native)
        call(j.repair_result, job=job, id=aid, attempt=1, input=native, failure=None,
             model=None, tool_reference='offline-saved-fixture-not-a-provider-call')
        bound = dict(config(400), input_sha256=j.digest(job / 'review' / f'{aid}.png'),
                     head_polygon=[[x + 4, y + 4] for x, y in config()['head_polygon']],
                     upper_body_bbox=[0, 0, 308, 404])
        bp = root / f'{aid}-layout.json'
        bp.write_text(json.dumps(bound), encoding='utf-8')
        call(j.portrait_layout, job=job, id=aid, layout=bp)
    m = j.load_job(job)[1]
    assert all(a.get('portrait_layout') for a in m['assets'][:4])
    assert sum(len(a.get('repair_attempts', [])) for a in m['assets']) == 2
    # Evidence tampering blocks acceptance, then the preserved mask is restored.
    mask_path = job / m['assets'][0]['portrait_layout']['head_mask_path']
    mask_bytes = mask_path.read_bytes()
    with Image.open(mask_path) as saved:
        mask_size = saved.size
    Image.new('L', mask_size).save(mask_path)
    invalid(j.review, job=job, id=['native'], decision='accept', note='Tampered head mask')
    mask_path.write_bytes(mask_bytes)
    call(j.review, job=job, id=['native', 'native_other', 'extract', 'complete'], decision='accept',
         note='Offline synthetic geometry, Alpha and source pixels verified; not a real-person identity test')
    invalid(j.portrait_layout, job=job, id='native', layout=layout_path)
    call(j.finalize, job=job)
    _, verified = call(j.verify, job=job)
    report = json.loads((job / 'delivery.json').read_text(encoding='utf-8'))
    assert report['local_pass'] == 2 and report['generated_pass'] == 2
    assert report['counts']['PASS'] == 4 and report['counts']['MANUAL'] == 2
    assert all(a['portrait_layout'] for a in report['assets'][:4])
    assert all(not a.get('pending_portrait') for a in j.load_job(job)[1]['assets'])
    invalid_plan = dict(plan, candidates=[dict(base, id='invalid', portrait='yes')])
    pp = root / 'invalid-plan.json'
    pp.write_text(json.dumps(invalid_plan), encoding='utf-8')
    try:
        call(j.build, plan=pp, job=root / 'invalid-job')
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid portrait flag accepted')
    assert not (root / 'invalid-job').exists()
    opaque = root / 'opaque.png'
    source.convert('RGB').save(opaque)
    fallback_plan = dict(plan, sources=[{'id': 's', 'path': str(opaque)}], candidates=[
        dict(base, id='fallback', repair_mode='complete'), dict(base, id='strict_fallback', repair_allowed=False)])
    fp = root / 'fallback-plan.json'
    fp.write_text(json.dumps(fallback_plan), encoding='utf-8')
    fallback_job = root / 'fallback-job'
    call(j.build, plan=fp, job=fallback_job)
    fallback = j.load_job(fallback_job)[1]['assets']
    assert fallback[0]['status'] == 'WAITING_REPAIR' and fallback[0]['repair_mode'] == 'complete'
    assert fallback[1]['status'] == 'MANUAL'
    return {'counts': report['counts'], 'verified_files': verified['verified_files'],
            'native_no_generation': True, 'interruption_recovered': True, 'separate_cli_workers': 2,
            'stale_layout_rejected': True, 'head_mask_tamper_rejected': True, 'portrait_fallback_scoped': True,
            'real_portrait_tested': False,
            'provider_called': False, 'saved_fixture_imports': 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('--output must be a new directory')
    with tempfile.TemporaryDirectory(prefix='portrait-offline-') as tmp:
        root = Path(tmp)
        report = {'geometry': geometry_checks(), 'workflow': workflow_checks(root), 'status': 'PASS'}
        if args.output:
            shutil.copytree(root, args.output)
            (args.output / 'portrait-test-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('PORTRAIT TEST PASS (offline synthetic silhouettes; no real face or provider call)')
        print(json.dumps(report))


if __name__ == '__main__':
    main()

"""Offline behavior checks for cached discovery and batch visual review.

Synthetic decisions exercise the transaction, not real visual acceptance.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image
import asset_job as j
from test_parallel import call, cli_batch, fixture, manifest


def write(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def rejected(function, **values):
    record = Path(values['job']) / 'manifest.json' if 'job' in values else None
    before = j.digest(record) if record and record.exists() else None
    try:
        call(function, **values)
    except (ValueError, FileExistsError):
        pass
    else:
        raise AssertionError('Invalid operation accepted')
    if before:
        assert j.digest(Path(values['job']) / 'manifest.json') == before


def decisions(root, batch, choices=None):
    return write(root / 'decisions.json', {'index_sha256': batch['index_sha256'], 'decisions': [
        {'id': aid, 'decision': (choices or {}).get(aid, 'accept'), 'note': 'Synthetic edge observation: ' + aid}
        for aid in batch['ids']]})


def batch_tests(root):
    for count, limit in [(1, 8), (8, 8), (12, 12), (13, 8)]:
        job = fixture(root, 'batch-' + str(count), count, route='A')
        result = call(j.review_sheet, job=job, batch_size=limit)[1]
        assert [aid for batch in result['batches'] for aid in batch['ids']] == [a['id'] for a in manifest(job)['assets']]
        assert len(result['batches']) == (count + limit - 1) // limit
        assert not list((job / 'assets').iterdir()), 'QC alone cannot PASS'
    job = root / 'batch-8' / 'job'
    batch = call(j.review_sheet, job=job, batch_size=8)[1]['batches'][0]
    path = decisions(root, batch, {'asset_01': 'reject', 'asset_02': 'detail'})
    original = json.loads(path.read_text())
    for bad in [dict(original, decisions=original['decisions'][:-1]),
                dict(original, decisions=[original['decisions'][0]] * 8),
                dict(original, index_sha256='stale')]:
        write(path, bad)
        rejected(j.review_batch, job=job, sheet=batch['index'], decisions=path)
    bad = copy.deepcopy(original)
    bad['decisions'][-1]['note'] = ' '
    write(path, bad)
    rejected(j.review_batch, job=job, sheet=batch['index'], decisions=path)
    write(path, original)
    call(j.review_batch, job=job, sheet=batch['index'], decisions=path)
    m = manifest(job)
    assert m['counts']['PASS'] == 6 and m['counts']['WAITING_REPAIR'] == 1 and m['counts']['REVIEW'] == 1
    assert m['assets'][2]['requires_detail_review']
    assert m['assets'][0]['visual_review']['mode'] == 'batch'
    call(j.review_batch, job=job, sheet=batch['index'], decisions=path)  # Safe idempotent replay.
    assert call(j.review_sheet, job=job)[1]['detail_ids'] == ['asset_02']
    call(j.review, job=job, id=['asset_02'], decision='accept', note='Synthetic enlarged edge inspected')
    assert manifest(job)['assets'][2]['visual_review']['mode'] == 'detail'
    for field in ['portrait', 'generated_repair', 'requires_detail_review']:
        a = dict(m['assets'][0], **{field: True})
        assert j.review_kind(a) == 'detail'
    assert j.review_kind(dict(m['assets'][0], extraction_method='bright-background')) == 'detail'
    assert j.review_kind(dict(m['assets'][0], metrics={'foreground_bbox': [0, 0, 10, 40]})) == 'detail'
    assert j.review_kind(dict(m['assets'][0], preview_sha256=None)) == 'detail'

    # Binding checks reject stale sources, review images, previews, indexes and sheets.
    for kind in ['candidates', 'review', 'previews', 'sheet', 'index']:
        job = fixture(root, 'stale-' + kind, 2, route='A')
        batch = call(j.review_sheet, job=job)[1]['batches'][0]
        path = decisions(root, batch)
        target = Path(batch['image'] if kind == 'sheet' else batch['index']) if kind in ('sheet', 'index') else (
            job / kind / ('asset_01.jpg' if kind == 'previews' else 'asset_01.png'))
        target.write_bytes(target.read_bytes() + b'changed')
        rejected(j.review_batch, job=job, sheet=batch['index'], decisions=path)
        assert not list((job / 'assets').iterdir())

    # Interrupt after the first move. Every pending decision must remain recoverable.
    job = fixture(root, 'interrupted', 3, route='A')
    batch = call(j.review_sheet, job=job)[1]['batches'][0]
    path = decisions(root, batch)
    replace, moves = Path.replace, []
    def stop(self, destination):
        if self.parent.name == 'review':
            moves.append(self)
            if len(moves) == 2:
                raise OSError('Simulated interruption')
        return replace(self, destination)
    with patch.object(Path, 'replace', stop):
        try:
            call(j.review_batch, job=job, sheet=batch['index'], decisions=path)
        except OSError:
            pass
        else:
            raise AssertionError('Interruption did not occur')
    assert all(a.get('pending_review') for a in manifest(job)['assets'])
    call(j.review_batch, job=job, sheet=batch['index'], decisions=path)
    assert manifest(job)['counts']['PASS'] == 3
    # Finalization reads each unchanged file only once, verify rereads them independently.
    reads, hash_file = [], j.hash_file
    def tracked(path):
        reads.append(Path(path).resolve())
        return hash_file(path)
    with patch.object(j, 'hash_file', tracked):
        call(j.finalize, job=job)
    assert len(reads) == len(set(reads)), 'Finalize repeated a full-file hash'
    reads.clear()
    with patch.object(j, 'hash_file', tracked), j.hash_scope():
        first = job / 'assets' / 'asset_00.png'
        j.digest(first)
        call(j.verify, job=job)
    assert reads.count(first.resolve()) == 2, 'Verify must bypass hash memo'
    job = fixture(root, 'batch-processes', 3, route='A')
    batch = call(j.review_sheet, job=job)[1]['batches'][0]
    path = decisions(root, batch)
    command = ['review-batch', '--sheet', batch['index'], '--decisions', str(path)]
    results = []
    cli_batch(job, [command, command], results)
    assert all(r['returncode'] == 0 for r in results)
    assert manifest(job)['counts']['PASS'] == 3


def hash_tests(root):
    path = root / 'hash.bin'
    path.write_bytes(b'aaaa')
    original = j.hash_file
    with patch.object(j, 'hash_file', wraps=original) as counted, j.hash_scope():
        old = j.digest(path)
        assert j.digest(path.parent / '.' / path.name) == old
        assert counted.call_count == 1
        path.write_bytes(b'bbbb')
        assert j.digest(path) != old and counted.call_count == 2
    old_stat = path.stat()
    with j.hash_scope():
        old = j.digest(path)
    path.write_bytes(b'cccc')
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    with j.hash_scope():
        assert j.digest(path) != old, 'Restored mtime must not hide cross-command changes'


def discovery_tests(root):
    folder = root / 'discovery'
    folder.mkdir()
    source = folder / 'original.png'
    Image.new('RGB', (1001, 1503), (246, 244, 238)).save(source)
    cache = folder / 'cache'
    opts = dict(input=[source], proxy_size=512, intent='extract red geometry', cache_dir=cache)
    inventory = folder / 'inventory.json'
    with patch.object(j, 'read_image', wraps=j.read_image) as counted:
        assert not call(j.inventory, output=inventory, **opts)[1]['cache_hit']
        assert counted.call_count == 1
    data = json.loads(inventory.read_text())
    assert data['sources'][0]['proxy_size'] == [341, 512]
    plan = copy.deepcopy(data)
    plan['coordinate_space'] = 'proxy'
    plan['discovery']['status'] = 'complete'
    plan['candidates'] = [{'id': 'square', 'source_id': 'source_001', 'label': 'square', 'route': 'A',
                           'reason': 'Synthetic discovery fixture', 'bbox': [20, 20, 200, 300],
                           'background_rgb': [246, 244, 238], 'foreground_points': [[40, 40]]}]
    raw, ready = write(folder / 'raw.json', plan), folder / 'ready.json'
    call(j.discovery_save, inventory=inventory, plan=raw, output=ready)
    converted = json.loads(ready.read_text())
    assert converted['candidates'][0]['bbox'] == [58, 58, 588, 881]
    assert converted['candidates'][0]['foreground_points'] == [[117, 117]]
    with patch.object(j, 'read_image', wraps=j.read_image) as counted:
        hit = call(j.inventory, output=folder / 'warm.json', **opts)[1]
        assert hit['cache_hit'] and hit['candidates'] == 1 and counted.call_count == 0
    bad_plan = copy.deepcopy(plan)
    bad_plan['discovery']['status'] = 'incomplete'
    write(raw, bad_plan)
    rejected(j.discovery_save, inventory=inventory, plan=raw, output=folder / 'incomplete.json')
    bad_plan = copy.deepcopy(plan)
    bad_plan['sources'][0]['path'] = str(folder / 'different.png')
    write(raw, bad_plan)
    rejected(j.discovery_save, inventory=inventory, plan=raw, output=folder / 'mismatch.json')
    write(raw, plan)
    proxy = Path(data['sources'][0]['proxy_path'])
    original_proxy = proxy.read_bytes()
    proxy.write_bytes(original_proxy + b'tampered')
    rejected(j.discovery_save, inventory=inventory, plan=raw, output=folder / 'bad-proxy.json')
    with patch.object(j, 'read_image', wraps=j.read_image) as counted:
        assert call(j.inventory, output=folder / 'rebuilt-proxy.json', **opts)[1]['cache_hit']
        assert counted.call_count == 1
    for i, override in enumerate([{'intent': 'extract portrait'}, {'proxy_size': 256}, {'refresh_discovery': True}]):
        assert not call(j.inventory, output=folder / f'miss-{i}.json', **dict(opts, **override))[1]['cache_hit']
    with patch.object(j, 'DISCOVERY_VERSION', j.DISCOVERY_VERSION + 1):
        assert not call(j.inventory, output=folder / 'version.json', **opts)[1]['cache_hit']
    saved = Path(data['discovery_cache']['directory']) / 'plan.json'
    payload = json.loads(saved.read_text())
    payload['plan']['candidates'][0]['label'] = 'corrupted without matching hash'
    write(saved, payload)
    assert not call(j.inventory, output=folder / 'corrupt.json', **opts)[1]['cache_hit']
    Image.new('RGB', (1001, 1503), (20, 30, 40)).save(source)
    assert not call(j.inventory, output=folder / 'changed.json', **opts)[1]['cache_hit']
    rejected(j.build, plan=ready, job=folder / 'stale-job', workers=5)
    assert not (folder / 'stale-job').exists()
    rejected(j.discovery_save, inventory=inventory, plan=raw, output=folder / 'stale-ready.json')
    # Source headers and decoded pixels share EXIF-transposed coordinate space.
    rotated = folder / 'rotated.jpg'
    im = Image.new('RGB', (300, 600))
    exif = Image.Exif()
    exif[274] = 6
    im.save(rotated, exif=exif)
    assert j.image_size(rotated) == j.read_image(rotated).size == (600, 300)


def decode_tests(root):
    with patch.object(j, 'read_image', wraps=j.read_image) as counted:
        job = fixture(root, 'decode', 5, route='A')
    assert counted.call_count == 1, 'Build should decode a shared source once'
    assert manifest(job)['counts']['REVIEW'] == 5


def main():
    with tempfile.TemporaryDirectory(prefix='asset-efficiency-') as tmp:
        root = Path(tmp)
        hash_tests(root)
        decode_tests(root)
        batch_tests(root)
        discovery_tests(root)
    print('EFFICIENCY TEST PASS (offline; synthetic review decisions, no provider call)')


if __name__ == '__main__':
    main()

"""Offline concurrency checks using threads and independent CLI processes."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import asset_job as j


def ns(**values):
    return argparse.Namespace(**values)


def call(function, **values):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        result = function(ns(**values))
    return result, json.loads(output.getvalue()) if output.getvalue().strip() else None


def fixture(root, name, count, workers=5, route='B'):
    folder = root / name
    folder.mkdir()
    source = Image.new('RGB', (80, 80), (246, 244, 238))
    ImageDraw.Draw(source).rectangle((20, 20, 60, 60), fill=(220, 80, 20))
    source.save(folder / 'source.png')
    candidate = {'source_id': 's', 'label': 'synthetic square', 'bbox': [0, 0, 80, 80],
                 'route': route, 'reason': 'Offline concurrency fixture',
                 'repair_mode': 'extract', 'repair_prompt': 'Isolate synthetic square',
                 'background_rgb': [246, 244, 238]}
    plan = {'discovery': {'provider': 'synthetic-offline-test', 'status': 'fixture',
                         'note': 'No image provider call; review decisions are fixtures'},
            'sources': [{'id': 's', 'path': str(folder / 'source.png')}],
            'candidates': [dict(candidate, id=f'asset_{i:02d}') for i in range(count)]}
    plan_path = folder / 'plan.json'
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    job = folder / 'job'
    assert call(j.build, plan=plan_path, job=job, workers=workers, processing='builtin-repair')[0] == 0
    return job


def manifest(job):
    return j.load_job(job)[1]


def assert_error(function, **values):
    before = j.digest(Path(values['job']) / 'manifest.json')
    try:
        call(function, **values)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid transition accepted')
    assert j.digest(Path(values['job']) / 'manifest.json') == before, 'Rejected operation changed manifest'


def import_args(job, aid, image, attempt=1):
    return dict(job=job, id=aid, attempt=attempt, input=image, failure=None,
                model=None, tool_reference='synthetic-offline-fixture')


def cli_batch(job, commands, report):
    # Delay the real manifest writer in each process. Without an OS lock this
    # widens the read/write race; with the lock the transaction stays atomic.
    script = (
        "import sys,time;sys.path.insert(0,sys.argv[1]);import asset_job as j;"
        "saved=j.save_manifest;"
        "j.save_manifest=lambda *a:(time.sleep(.12),saved(*a))[1];"
        "sys.argv=sys.argv[2:];raise SystemExit(j.main())"
    )
    processes = []
    env = dict(os.environ, PYTHONUTF8='1')
    for command in commands:
        argv = [sys.executable, '-c', script, str(Path(j.__file__).parent),
                'asset_job.py', command[0], '--job', str(job), *map(str, command[1:])]
        processes.append(subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, encoding='utf-8', env=env))
    results = []
    try:
        for process, command in zip(processes, commands):
            stdout, stderr = process.communicate(timeout=45)
            results.append({'command': list(map(str, command)), 'returncode': process.returncode,
                            'stdout': stdout.strip(), 'stderr': stderr.strip()})
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    report.extend(results)
    return results


def thread_checks(root):
    observations = []
    for count, workers in [(1, 5), (2, 5), (5, 5), (8, 5), (7, 3), (8, 7), (3, 1)]:
        effective = min(count, workers)
        barrier = threading.Barrier(effective, timeout=15)
        mutex = threading.Lock()
        active = maximum = entered = 0
        original = j.build_candidate

        def overlapping(job, candidate):
            nonlocal active, maximum, entered
            with mutex:
                ticket = entered
                entered += 1
                active += 1
                maximum = max(maximum, active)
            try:
                if ticket < effective:
                    barrier.wait()
                time.sleep(.02)
                return original(job, candidate)
            finally:
                with mutex:
                    active -= 1

        started = time.monotonic()
        with patch.object(j, 'build_candidate', side_effect=overlapping):
            parallel = fixture(root, f'threads-{count}-{workers}', count, workers, 'A')
        duration = time.monotonic() - started
        assert maximum == effective, (count, workers, maximum)
        assert entered == count and active == 0
        m = manifest(parallel)
        assert m['max_parallel'] == workers
        assert m['processing'] == 'builtin-repair'
        assert [a['id'] for a in m['assets']] == [f'asset_{i:02d}' for i in range(count)]
        assert m['counts']['REVIEW'] == count and not m['counts']['ERROR']
        serial = fixture(root, f'serial-{count}-{workers}', count, 1, 'A')
        for asset in m['assets']:
            name = asset['id'] + '.png'
            assert np.array_equal(np.asarray(j.read_image(parallel / 'review' / name)),
                                  np.asarray(j.read_image(serial / 'review' / name)))
        observations.append({'candidates': count, 'requested_workers': workers,
                             'observed_peak': maximum, 'seconds': round(duration, 3),
                             'pixels_match_serial': True})
    # Namespace callers and CLI callers must both reject invalid worker counts.
    plan_path = root / 'threads-1-5' / 'plan.json'
    for workers in (0, -1, True, 1.5):
        job = root / f'invalid-{workers}'
        try:
            call(j.build, plan=plan_path, job=job, workers=workers)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid workers accepted: {workers}')
        assert not job.exists(), 'Invalid configuration created a partial job'
    return observations


def process_checks(root, native, report):
    job = fixture(root, 'process-races', 6, 2)
    ids = [a['id'] for a in manifest(job)['assets']]
    results = cli_batch(job, [('repair-start', '--id', aid, '--worker', 'race-worker') for aid in ids], report)
    assert sum(r['returncode'] == 0 for r in results) == 2, results
    assert all(r['returncode'] in (0, 1) for r in results)
    m = manifest(job)
    started = [a['id'] for a in m['assets'] if a['status'] == 'REPAIRING']
    waiting = [a['id'] for a in m['assets'] if a['status'] == 'WAITING_REPAIR']
    assert len(started) == 2 and len(waiting) == 4
    for aid in started:
        attempt = next(a for a in m['assets'] if a['id'] == aid)['repair_attempts']
        assert len(attempt) == 1 and attempt[0]['worker'] == 'race-worker'
    queue = call(j.repair_queue, job=job)[1]
    assert queue['capacity'] == {'max_parallel': 2, 'active': 2, 'available_slots': 0}
    assert len(queue['active_tasks']) == 2 and len(queue['tasks']) == 4
    for result in results:
        if result['returncode'] == 0:
            output = json.loads(result['stdout'])
            aid = output['id']
            assert output['attempt'] == 1
            assert output['referenced_image_paths'] == [str(job / 'candidates' / f'{aid}.png')]

    # Unknown outcomes reserve the slot, and recovery imports without redispatch.
    assert call(j.repair_result, **dict(import_args(job, started[0], None),
                                      failure='Synthetic unknown provider outcome'))[0] == 2
    assert_error(j.repair_start, job=job, id=waiting[0])
    status = call(j.status, job=job)[1]
    assert status['capacity']['active'] == 2 and status['capacity']['available_slots'] == 0
    assert call(j.repair_result, **import_args(job, started[0], native[0]))[0] == 0
    call(j.repair_start, job=job, id=waiting[0])
    other_ids = [waiting[0], started[1]]  # reverse start order; never bind by return order
    results = cli_batch(job, [('repair-result', '--id', aid, '--attempt', 1, '--input', image)
                             for aid, image in zip(other_ids, native[1:])], report)
    assert all(r['returncode'] == 0 for r in results), results
    for aid, image in zip([started[0], *other_ids], native):
        assert np.array_equal(np.asarray(j.read_image(image)), np.asarray(j.read_image(job / 'review' / f'{aid}.png')))
    assert manifest(job)['counts']['REVIEW'] == 3
    results = cli_batch(job, [('review', '--id', aid, '--decision', 'accept',
                              '--note', 'Synthetic source/alpha/light/dark fixture checked')
                             for aid in [started[0], *other_ids]], report)
    assert all(r['returncode'] == 0 for r in results), results
    assert manifest(job)['counts']['PASS'] == 3
    assert all(len(a.get('repair_attempts', [])) == 1 for a in manifest(job)['assets'] if a['status'] == 'PASS')

    # A late attempt must fail before an artifact write or a failure-state update.
    stale = waiting[1]
    assert_error(j.repaired, job=job, id=stale, input=job.parent / 'source.png',
                 background=[246, 244, 238], point=[])
    assert not (job / 'repaired' / f'{stale}.png').exists(), 'Legacy import bypassed attempt registration'
    call(j.repair_start, job=job, id=stale)
    call(j.repair_result, **import_args(job, stale, native[0]))
    call(j.review, job=job, id=[stale], decision='reject', note='Synthetic retry exercise')
    call(j.repair_start, job=job, id=stale)
    assert_error(j.repair_result, **import_args(job, stale, native[1], attempt=1))
    assert_error(j.repair_result, **dict(import_args(job, stale, None, attempt=1), failure='Late failure'))
    assert_error(j.repair_result, **import_args(job, stale, native[1], attempt=None))
    assert not (job / 'repaired' / f'{stale}-attempt-2.png').exists()
    call(j.repair_result, **import_args(job, stale, native[1], attempt=2))
    call(j.review, job=job, id=[stale], decision='accept', note='Synthetic second attempt checked')
    assert_error(j.repair_start, job=job, id=stale)

    exhausted = waiting[2]
    for number in (1, 2):
        call(j.repair_start, job=job, id=exhausted)
        assert call(j.repair_result, **import_args(job, exhausted, job.parent / 'source.png', number))[0] == 1
    assert next(a for a in manifest(job)['assets'] if a['id'] == exhausted)['status'] == 'MANUAL'
    assert_error(j.repair_start, job=job, id=exhausted)
    last = waiting[3]
    call(j.repair_start, job=job, id=last)
    call(j.repair_result, **import_args(job, last, native[2]))
    call(j.review, job=job, id=[last], decision='accept', note='Synthetic final fixture checked')
    call(j.finalize, job=job)
    index = json.loads((job / 'checksums.json').read_text(encoding='utf-8'))
    assert '.job.lock' not in index and (job / '.job.lock').exists()
    call(j.status, job=job)
    call(j.repair_queue, job=job)
    call(j.verify, job=job)
    assert manifest(job)['counts']['PASS'] == 5 and manifest(job)['counts']['MANUAL'] == 1

    duplicate = fixture(root, 'duplicate-id', 2, 2)
    results = cli_batch(duplicate, [('repair-start', '--id', 'asset_00')] * 2, report)
    assert sorted(r['returncode'] for r in results) == [0, 1], results
    m = manifest(duplicate)
    assert len(m['assets'][0]['repair_attempts']) == 1 and m['counts']['REPAIRING'] == 1
    assert not m['assets'][1].get('repair_attempts')

    five = fixture(root, 'five-slot-capacity', 7, 5)
    results = cli_batch(five, [('repair-start', '--id', f'asset_{i:02d}') for i in range(7)], report)
    assert sum(r['returncode'] == 0 for r in results) == 5, results
    capacity = call(j.status, job=five)[1]['capacity']
    assert capacity == {'max_parallel': 5, 'active': 5, 'available_slots': 0}
    assert sum(len(a.get('repair_attempts', [])) for a in manifest(five)['assets']) == 5

    legacy = fixture(root, 'legacy', 2, 5)
    with j.job_lock(legacy):
        m = manifest(legacy)
        m.pop('max_parallel')
        m.pop('processing')
        j.save_manifest(legacy, m)
    legacy_queue = call(j.repair_queue, job=legacy)[1]
    assert legacy_queue['capacity']['max_parallel'] == 1
    assert legacy_queue['processing'] == 'builtin-repair'
    assert call(j.status, job=legacy)[1]['processing'] == 'builtin-repair'
    call(j.repair_start, job=legacy, id='asset_00')
    assert_error(j.repair_start, job=legacy, id='asset_01')
    args = import_args(legacy, 'asset_00', native[0])
    args.pop('attempt')
    call(j.repair_result, **args)
    call(j.review, job=legacy, id=['asset_00'], decision='accept', note='Synthetic legacy checked')
    call(j.repair_start, job=legacy, id='asset_01')
    args = import_args(legacy, 'asset_01', native[1])
    args.pop('attempt')
    call(j.repair_result, **args)
    call(j.review, job=legacy, id=['asset_01'], decision='accept', note='Synthetic legacy checked')
    call(j.finalize, job=legacy)
    call(j.verify, job=legacy)
    assert json.loads((legacy / 'delivery.json').read_text(encoding='utf-8'))['processing'] == 'builtin-repair'
    assert 'processing' not in manifest(legacy), 'Reading an old job must not silently migrate its mode'

    legacy_import = fixture(root, 'legacy-manual-import', 1, 5)
    with j.job_lock(legacy_import):
        m = manifest(legacy_import)
        m.pop('max_parallel')
        m.pop('processing')
        j.save_manifest(legacy_import, m)
    call(j.repaired, job=legacy_import, id='asset_00', input=legacy_import.parent / 'source.png',
         background=[246, 244, 238], point=[])
    assert manifest(legacy_import)['counts']['REVIEW'] == 1
    call(j.review, job=legacy_import, id=['asset_00'], decision='accept', note='Synthetic legacy manual import checked')
    call(j.finalize, job=legacy_import)
    call(j.verify, job=legacy_import)


def profile_checks(root, native):
    folder = root / 'local-first-default'
    folder.mkdir()
    source = Image.new('RGB', (80, 80), (246, 244, 238))
    ImageDraw.Draw(source).rectangle((20, 20, 60, 60), fill=(220, 80, 20))
    source.save(folder / 'source.png')
    candidate = {'source_id': 's', 'label': 'synthetic square', 'bbox': [0, 0, 80, 80],
                 'route': 'A', 'reason': 'Offline local-first fixture'}
    plan = {'discovery': {'provider': 'synthetic-offline-test', 'status': 'fixture',
                         'note': 'No image provider call; completion is a saved fixture'},
            'sources': [{'id': 's', 'path': str(folder / 'source.png')}],
            'candidates': [dict(candidate, id='local_good', background_rgb=[246, 244, 238]),
                           dict(candidate, id='local_failed', background_rgb=[0, 0, 0]),
                           dict(candidate, id='auto_complete_failed', repair_mode='complete', background_rgb=[0, 0, 0]),
                           dict(candidate, id='plain_extract', route='B', repair_prompt='Separate square'),
                           dict(candidate, id='completion', route='B', repair_mode='complete',
                                repair_prompt='Complete the specified synthetic square'),
                           dict(candidate, id='strict_completion', route='B', repair_mode='complete',
                                repair_allowed=False, repair_prompt='Complete square only when allowed')]}
    plan_path = folder / 'plan.json'
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    job = folder / 'job'
    assert call(j.build, plan=plan_path, job=job, workers=5)[0] == 0
    m = manifest(job)
    assert m['processing'] == 'local-first'
    assert m['plan_sha256'] == j.digest(plan_path) == j.digest(job / 'plan.json')
    states = {a['id']: a['status'] for a in m['assets']}
    assert states == {'local_good': 'REVIEW', 'local_failed': 'MANUAL', 'auto_complete_failed': 'MANUAL', 'plain_extract': 'MANUAL',
                      'completion': 'WAITING_REPAIR', 'strict_completion': 'MANUAL'}, states
    assert not any(a.get('repair_attempts') for a in m['assets'])
    for aid in ('local_failed', 'auto_complete_failed', 'plain_extract', 'strict_completion'):
        assert_error(j.repair_start, job=job, id=aid)
    queue = call(j.repair_queue, job=job)[1]
    assert queue['processing'] == call(j.status, job=job)[1]['processing'] == 'local-first'
    queued = queue['tasks']
    assert [task['id'] for task in queued] == ['completion']
    call(j.repair_start, job=job, id='completion')
    assert call(j.repair_result, **import_args(job, 'completion', native))[0] == 0
    call(j.review, job=job, id=['local_good', 'completion'], decision='accept',
         note='Synthetic local-first saved fixture; no provider call')
    m = manifest(job)
    assert sum(len(a.get('repair_attempts', [])) for a in m['assets']) == 1
    assert m['counts']['PASS'] == 2 and m['counts']['MANUAL'] == 4
    call(j.finalize, job=job)
    call(j.verify, job=job)
    assert json.loads((job / 'delivery.json').read_text(encoding='utf-8'))['processing'] == 'local-first'
    invalid = folder / 'invalid-processing'
    try:
        call(j.build, plan=plan_path, job=invalid, processing='unexpected-profile')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown processing mode accepted')
    assert not invalid.exists(), 'Unknown processing mode created a job'
    return {'default_processing': 'local-first', 'states_after_build': states,
            'registered_requests': 1, 'provider_called': False,
            'plan_sha256_preserved': True, 'invalid_mode_rejected_before_job_creation': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('--output must be a new directory')
    with tempfile.TemporaryDirectory(prefix='asset-parallel-test-') as temporary:
        root = Path(temporary)
        native = []
        for number, color in enumerate([(200, 70, 10, 255), (30, 100, 190, 255), (80, 160, 40, 255)]):
            image = Image.new('RGBA', (100, 90), (0, 0, 0, 0))
            drawing = ImageDraw.Draw(image)
            drawing.rectangle((20, 20, 80, 70), fill=color)
            drawing.line((19, 20, 19, 70), fill=(*color[:3], 128))
            image_path = root / f'native-{number}.png'
            image.save(image_path)
            native.append(image_path)
        thread_report = thread_checks(root)
        processes = []
        process_checks(root, native, processes)
        profile_report = profile_checks(root, native[0])
        summary = {'provider_called': False, 'thread_checks': thread_report,
                   'process_commands': processes, 'processing_profile': profile_report, 'status': 'PASS'}
        (root / 'parallel-test-report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.output:
            shutil.copytree(root, args.output.resolve())
    print('PARALLEL WORKFLOW TEST PASS (offline; threads and independent CLI processes; no provider call)')
    print(json.dumps(thread_report))


if __name__ == '__main__':
    main()

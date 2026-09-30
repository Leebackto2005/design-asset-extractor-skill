"""Offline behavioral checks. No image-model call is made by this test."""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
import sys
import contextlib
import io
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import asset_job as j


def ns(**kw):
    return argparse.Namespace(**kw)


def result_args(**kw):
    _, manifest = j.load_job(kw['job'])
    asset = next(a for a in manifest['assets'] if a['id'] == kw['id'])
    return ns(attempt=asset['repair_attempts'][-1]['number'], **kw)


def main():
    card = Image.new('RGBA', (40, 30), (123, 80, 20, 255))
    padded = j.local_extract(card, 'crop', 8)
    assert np.array_equal(np.asarray(padded)[8:38, 8:48], np.asarray(card))
    assert padded.getpixel((0, 0))[3] == 0
    checker = Image.new('RGB', (100, 120), 'white')
    d = ImageDraw.Draw(checker)
    for y in range(0, 120, 10):
        for x in range(0, 100, 10):
            if (x // 10 + y // 10) % 2:
                d.rectangle((x, y, x+9, y+9), fill=(210, 210, 210))
    d.ellipse((25, 20, 75, 100), fill=(25, 30, 35))
    extracted = j.local_extract(checker.convert('RGBA'), 'bright-background')
    assert extracted.getpixel((50, 60)) == (25, 30, 35, 255)
    assert extracted.getpixel((5, 5))[3] == 0 and extracted.getpixel((15, 5))[3] == 0
    j.inspect(extracted)
    # A transparent border must never conceal a clipped or unremoved crop.
    with tempfile.TemporaryDirectory(prefix='asset-local-gates-') as tmp:
        root = Path(tmp)
        clipped = Image.new('RGB', (80, 80), (246, 244, 238))
        ImageDraw.Draw(clipped).rectangle((0, 20, 60, 60), fill='orange')
        clipped.save(root / 'clipped.png')
        checker.save(root / 'checker.png')
        Image.new('RGB', (80, 80), 'white').save(root / 'light.png')
        local = {'id': 'clipped', 'source_id': 'clipped', 'label': 'square', 'bbox': [0, 0, 80, 80],
                 'route': 'A', 'reason': 'Synthetic local gate', 'background_rgb': [246, 244, 238]}
        plan = {'sources': [{'id': name, 'path': str(root / f'{name}.png')}
                            for name in ('clipped', 'checker', 'light')],
                'candidates': [local, dict(local, id='strict_clipped', repair_allowed=False),
                               dict(local, id='bright_failed', source_id='light', extraction_method='bright-background'),
                               dict(local, id='bright_rejected', source_id='checker', bbox=[0, 0, 100, 120],
                                    extraction_method='bright-background'),
                               dict(local, id='photo', bbox=[20, 20, 61, 61], extraction_method='crop')]}
        pp = root / 'plan.json'
        pp.write_text(json.dumps(plan), encoding='utf-8')
        job = root / 'job'
        j.build(ns(plan=pp, job=job, processing='builtin-repair'))
        _, m = j.load_job(job)
        assert m['assets'][0]['status'] == 'WAITING_REPAIR', 'Padding hid a clipped target'
        assert m['assets'][1]['status'] == 'MANUAL', 'Strict clipping must stay local/manual'
        assert m['assets'][2]['status'] == 'MANUAL', 'Bright-background failure must not generate'
        j.review(ns(job=job, id=['bright_rejected'], decision='reject', note='Synthetic edge rejection'))
        _, m = j.load_job(job)
        assert m['assets'][3]['status'] == 'MANUAL' and not m['assets'][3].get('repair_attempts')
        photo = j.read_image(job / 'review' / 'photo.png')
        assert np.array_equal(np.asarray(photo)[12:-12, 12:-12], np.asarray(clipped.convert('RGBA'))[20:61, 20:61])
    with tempfile.TemporaryDirectory(prefix='asset-builtin-test-') as tmp:
        root = Path(tmp)
        source = Image.new('RGB', (80, 80), (246, 244, 238))
        ImageDraw.Draw(source).rectangle((20, 20, 60, 60), fill='orange')
        source.save(root / 'source.png')
        j.inventory(ns(input=[str(root)], output=root / 'inventory.json'))
        assert len(json.loads((root / 'inventory.json').read_text(encoding='utf-8'))['sources']) == 1
        b = {'id': 'b', 'source_id': 's', 'label': 'square', 'bbox': [0, 0, 80, 80],
             'route': 'B', 'reason': 'model repair', 'repair_mode': 'extract', 'repair_prompt': 'Isolate square'}
        plan = {'discovery': {'provider': 'synthetic-offline-test', 'status': 'fixture', 'note': 'No image model called; review decisions are test fixtures'},
                'sources': [{'id': 's', 'path': str(root / 'source.png')}],
                'candidates': [b, dict(b, id='opaque'), dict(b, id='blocked'),
                               dict(b, id='a', route='A', background_rgb=[0, 0, 0]), dict(b, id='c', route='C')]}
        (root / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        job = root / 'job'
        assert j.build(ns(plan=root / 'plan.json', job=job, processing='builtin-repair')) == 0
        _, m = j.load_job(job)
        assert m['assets'][3]['initial_route'] == 'AUTO' and m['assets'][3]['status'] == 'WAITING_REPAIR'
        assert m['assets'][4]['status'] == 'MANUAL'
        native = Image.new('RGBA', (100, 90), (0, 0, 0, 0))
        draw = ImageDraw.Draw(native)
        draw.rectangle((20, 20, 80, 70), fill=(200, 70, 10, 255))
        draw.rectangle((19, 20, 19, 70), fill=(200, 70, 10, 128))
        native.save(root / 'native.png')
        for attempt in range(2):
            j.repair_start(ns(job=job, id='b'))
            try:
                j.repair_start(ns(job=job, id='b'))
            except ValueError:
                pass
            else:
                raise AssertionError('Duplicate dispatch allowed')
            assert j.repair_result(result_args(job=job, id='b', failure=None, input=root / 'native.png', model=None, tool_reference=None)) == 0
            assert np.array_equal(np.asarray(native), np.asarray(j.read_image(job / 'review' / 'b.png')))
            j.review(ns(job=job, id=['b'], decision='reject' if attempt == 0 else 'accept', note='Synthetic test observation; no model call'))
        _, m = j.load_job(job)
        a = m['assets'][0]
        assert a['status'] == 'PASS' and len(a['repair_attempts']) == 2
        assert a['repair_attempts'][-1]['model'] is None
        assert (job / 'rejected' / 'b-attempt-1.png').exists()
        assert len(list((job / 'repaired').glob('b-attempt-*.png'))) == 2
        for _ in range(2):
            j.repair_start(ns(job=job, id='opaque'))
            assert j.repair_result(result_args(job=job, id='opaque', failure=None, input=root / 'source.png', model=None, tool_reference=None)) == 1
        _, m = j.load_job(job)
        assert m['assets'][1]['status'] == 'MANUAL'
        j.repair_start(ns(job=job, id='blocked'))
        assert j.repair_result(result_args(job=job, id='blocked', failure='Unknown external result', input=None)) == 2
        assert j.repair_result(result_args(job=job, id='blocked', failure=None, input=root / 'native.png', model=None, tool_reference='recovered')) == 0
        _, m = j.load_job(job)
        assert len(m['assets'][2]['repair_attempts']) == 1
        assert m['counts']['PASS'] == 1 and not (job / 'assets' / 'opaque.png').exists()
        assert (job / 'manual' / 'opaque.json').exists(), 'Retry exhaustion must include handoff'
        try:
            j.finalize(ns(job=job))
        except ValueError:
            pass
        else:
            raise AssertionError('Unresolved job delivered')
        j.review(ns(job=job, id=['blocked'], decision='accept', note='Synthetic alpha checked'))
        j.repair_start(ns(job=job, id='a'))
        # Inject an interruption after the received artifact has been saved.
        with patch.object(j, 'preview', side_effect=OSError('simulated preview interruption')):
            try:
                j.repair_result(result_args(job=job, id='a', failure=None, input=root / 'native.png', model=None, tool_reference='offline'))
            except OSError:
                pass
            else:
                raise AssertionError('Fault injection did not execute')
        changed = native.copy()
        changed.putpixel((30, 30), (1, 2, 3, 255))
        changed.save(root / 'changed.png')
        try:
            j.repair_result(result_args(job=job, id='a', failure=None, input=root / 'changed.png', model=None, tool_reference='offline'))
        except ValueError:
            pass
        else:
            raise AssertionError('A different image replaced existing attempt')
        j.repair_result(result_args(job=job, id='a', failure=None, input=root / 'native.png', model=None, tool_reference='offline'))
        j.review(ns(job=job, id=['a'], decision='accept', note='Synthetic recovery checked'))
        j.finalize(ns(job=job))
        j.verify(ns(job=job))
        _, m = j.load_job(job)
        assert len(m['assets'][3]['repair_attempts']) == 1, 'Recovery must not dispatch again'
        # Strict extraction never creates an un-runnable repair queue.
        strict = dict(b, id='strict', repair_allowed=False)
        plan['candidates'] = [strict, dict(strict, id='failed', route='A', background_rgb=[0, 0, 0]),
                              dict(strict, id='rejected', route='A', background_rgb=[246, 244, 238])]
        (root / 'strict.json').write_text(json.dumps(plan), encoding='utf-8')
        strict_job = root / 'strict-job'
        j.build(ns(plan=root / 'strict.json', job=strict_job, processing='builtin-repair'))
        j.review(ns(job=strict_job, id=['rejected'], decision='reject', note='Strict synthetic rejection'))
        _, m = j.load_job(strict_job)
        assert m['counts']['MANUAL'] == 3
        assert len(list((strict_job / 'manual').glob('*.json'))) == 3
        j.finalize(ns(job=strict_job))
        j.verify(ns(job=strict_job))
        for suffix in ('png', 'md', 'json'):
            handoff = strict_job / 'manual' / f'strict.{suffix}'
            saved = handoff.read_bytes()
            handoff.unlink()
            try:
                j.finalize(ns(job=strict_job))
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f'Incomplete manual handoff delivered without {suffix}')
            finally:
                handoff.write_bytes(saved)
        # An interrupted accept/reject resumes the same recorded decision.
        plan['candidates'] = [dict(strict, id='accept_resume', route='A', background_rgb=[246, 244, 238]),
                              dict(strict, id='reject_resume', route='A', background_rgb=[246, 244, 238])]
        (root / 'resume.json').write_text(json.dumps(plan), encoding='utf-8')
        resume_job = root / 'review-resume'
        j.build(ns(plan=root / 'resume.json', job=resume_job, processing='builtin-repair'))
        for aid, decision in [('accept_resume', 'accept'), ('reject_resume', 'reject')]:
            original_save = j.save_manifest

            def interrupt_after_move(job_path, manifest):
                if not (job_path / 'review' / f'{aid}.png').exists():
                    raise OSError('simulated review interruption')
                return original_save(job_path, manifest)

            args = ns(job=resume_job, id=[aid], decision=decision, note='Synthetic review recovery')
            with patch.object(j, 'save_manifest', side_effect=interrupt_after_move):
                try:
                    j.review(args)
                except OSError:
                    pass
                else:
                    raise AssertionError('Review fault injection did not execute')
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                j.status(ns(job=resume_job))
            action = next(a for a in json.loads(captured.getvalue())['actions'] if a['id'] == aid)
            assert action['pending_review']['decision'] == decision
            try:
                j.review(ns(job=resume_job, id=[aid], decision='reject' if decision == 'accept' else 'accept',
                            note=args.note))
            except ValueError:
                pass
            else:
                raise AssertionError('An interrupted review changed its decision')
            j.review(args)
        j.finalize(ns(job=resume_job))
        j.verify(ns(job=resume_job))
        if len(sys.argv) == 3 and sys.argv[1] == '--output':
            destination = Path(sys.argv[2]).resolve()
            shutil.copytree(job, destination / 'synthetic-workflow')
            shutil.copytree(strict_job, destination / 'strict-workflow')
            shutil.copytree(resume_job, destination / 'review-resume')
        (job / 'assets' / 'b.png').write_bytes(b'changed')
        try:
            j.verify(ns(job=job))
        except ValueError:
            pass
        else:
            raise AssertionError('Changed delivery passed verification')
    print('BUILTIN WORKFLOW TEST PASS (offline; no provider call)')


if __name__ == '__main__':
    main()

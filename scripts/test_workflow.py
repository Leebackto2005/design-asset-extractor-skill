"""Offline behavioral checks. No image-model call is made by this test."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import asset_job as j


def ns(**kw):
    return argparse.Namespace(**kw)


def main():
    with tempfile.TemporaryDirectory(prefix='asset-builtin-test-') as tmp:
        root = Path(tmp)
        source = Image.new('RGB', (80, 80), (246, 244, 238))
        ImageDraw.Draw(source).rectangle((20, 20, 60, 60), fill='orange')
        source.save(root / 'source.png')
        j.inventory(ns(input=[str(root)], output=root / 'inventory.json'))
        assert len(json.loads((root / 'inventory.json').read_text(encoding='utf-8'))['sources']) == 1
        b = {'id': 'b', 'source_id': 's', 'label': 'square', 'bbox': [0, 0, 80, 80],
             'route': 'B', 'reason': 'model repair', 'repair_mode': 'extract', 'repair_prompt': 'Isolate square'}
        plan = {'sources': [{'id': 's', 'path': str(root / 'source.png')}],
                'candidates': [b, dict(b, id='opaque'), dict(b, id='blocked'),
                               dict(b, id='a', route='A', background_rgb=[0, 0, 0]), dict(b, id='c', route='C')]}
        (root / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        job = root / 'job'
        assert j.build(ns(plan=root / 'plan.json', job=job)) == 0
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
            assert j.repair_result(ns(job=job, id='b', failure=None, input=root / 'native.png', model=None, tool_reference=None)) == 0
            assert np.array_equal(np.asarray(native), np.asarray(j.read_image(job / 'review' / 'b.png')))
            j.review(ns(job=job, id=['b'], decision='reject' if attempt == 0 else 'accept', note='test observation'))
        _, m = j.load_job(job)
        a = m['assets'][0]
        assert a['status'] == 'PASS' and len(a['repair_attempts']) == 2
        assert a['repair_attempts'][-1]['model'] is None
        assert (job / 'rejected' / 'b-attempt-1.png').exists()
        assert len(list((job / 'repaired').glob('b-attempt-*.png'))) == 2
        for _ in range(2):
            j.repair_start(ns(job=job, id='opaque'))
            assert j.repair_result(ns(job=job, id='opaque', failure=None, input=root / 'source.png', model=None, tool_reference=None)) == 1
        _, m = j.load_job(job)
        assert m['assets'][1]['status'] == 'MANUAL'
        j.repair_start(ns(job=job, id='blocked'))
        assert j.repair_result(ns(job=job, id='blocked', failure='Unknown external result', input=None)) == 2
        assert j.repair_result(ns(job=job, id='blocked', failure=None, input=root / 'native.png', model=None, tool_reference='recovered')) == 0
        _, m = j.load_job(job)
        assert len(m['assets'][2]['repair_attempts']) == 1
        assert m['counts']['PASS'] == 1 and not (job / 'assets' / 'opaque.png').exists()
    print('BUILTIN WORKFLOW TEST PASS (offline; no provider call)')


if __name__ == '__main__':
    main()

"""
Copy the ear model into api/vendor, so the hosted API can be deployed from the
api/ folder alone. Vercel builds the API from that folder and cannot see the
rest of the repository, and building it from the repository root does not fit
its function size limit once the page's dependencies are installed there.

Run after changing any of the files listed below:

    python tools/sync_model.py

A test (api/server/tests/test_app.py) fails while the copy is out of date.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'api' / 'vendor'
FILES = [
    'auditory_periphery.py',
    'cochlea/gammatone_frame.py',
    'haircell/inner_hair_cell.py',
    'neuron_models/spike_timing.py',
    'reconstruction/decode_spike_times.py',
]


def main():
    for name in FILES:
        target = VENDOR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        package = target.parent / '__init__.py'
        if target.parent != VENDOR and not package.exists():
            package.write_text('', encoding='utf-8')
        print('copied', name)


if __name__ == '__main__':
    main()

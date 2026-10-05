r"""
Build the demo the page opens with, from the last `recover.py` run: the four versions of the clip and what the strip chart draws.

Writes public/demo/. Run after `python recover.py --seconds 10`:

    python tools/build_demo.py
"""
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'api'))

from server.showcase import build_showcase  # noqa: E402

OUTPUT = ROOT / 'output'
DEMO = ROOT / 'public' / 'demo'
# Name the page uses -> file recover.py writes.
AUDIO = {
    'original': 'original',
    'regenerated': 'pathH_full_ear',
    'rate-code': 'pathC_neural_spikes',
    'envelope': 'pathB_envelope_vocoder',
}


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    for name, source in AUDIO.items():
        shutil.copyfile(OUTPUT / f'{source}.wav', DEMO / f'{name}.wav')
    x, fs = sf.read(OUTPUT / 'original.wav')
    showcase = build_showcase(x, fs)
    target = DEMO / 'showcase.json'
    target.write_text(json.dumps(showcase, separators=(',', ':')), encoding='utf-8')
    print(f"{DEMO}: {showcase['duration']:.1f} s, {len(showcase['raster'])} fibers, "
          f"{target.stat().st_size / 1e6:.2f} MB of chart data")


if __name__ == '__main__':
    main()

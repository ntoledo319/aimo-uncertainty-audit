"""Reconstruct entry from the pinned upstream source, without training/inference.

Use an existing upstream checkout with --upstream, or fetch only the eight needed
files with --download. No package installation or competition API call is made.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import urllib.request
import zipfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--upstream', type=Path)
    group.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory; existing paths are never overwritten')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / 'provenance.json').read_text())
    if args.output.exists():
        parser.error('output path already exists')
    args.output.mkdir(parents=True)
    solution = args.output / 'solution'
    solution.mkdir()
    for name, record in provenance['files'].items():
        if args.download:
            url = ('https://raw.githubusercontent.com/aimo-interp/getting-started/'
                   + provenance['commit'] + '/' + record['upstream_path'])
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        else:
            data = (args.upstream / record['upstream_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['upstream_sha256']:
            raise ValueError(f'Upstream checksum mismatch: {name}')
        dest = solution / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    subprocess.run(['patch', '--batch', '--forward', '-p1', '-i',
                    str(root / 'submission.patch')], cwd=solution, check=True)
    for name, record in provenance['files'].items():
        data = (solution / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['submitted_sha256']:
            raise ValueError(f'Reconstruction mismatch: {name}')
    for small in (False, True):
        filename = ('solution-uncertainty-hybrid-small.zip' if small
                    else 'solution-uncertainty-hybrid.zip')
        with zipfile.ZipFile(args.output / filename, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in sorted(provenance['files']):
                z.write(solution / name, name)
            if small:
                z.writestr('small.txt', '')
    print('Verified every reconstructed source/artifact byte against the frozen entry.')
    print('ZIP bytes may differ due to ZIP metadata; entry contents are verified.')

if __name__ == '__main__':
    main()

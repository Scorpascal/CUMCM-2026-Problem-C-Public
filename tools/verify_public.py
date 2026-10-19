"""Read-only SHA-256 verification of a public derivative, not the private freeze."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest_path = root / 'PUBLIC_MANIFEST.json'
    if manifest_path.is_symlink():
        return ['symlink: PUBLIC_MANIFEST.json']
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return [f'manifest unavailable: {exc}']
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), list):
        return ['unsupported manifest format']
    errors, expected = [], set()
    for entry in manifest['files']:
        name = entry.get('path') if isinstance(entry, dict) else None
        if (not isinstance(name, str) or not name or '\\' in name
                or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                or name != PurePosixPath(name).as_posix()):
            errors.append(f'unsafe path: {name!r}')
            continue
        if name in expected:
            errors.append(f'duplicate: {name}')
            continue
        expected.add(name)
        path = root / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != root and root in p.parents):
            errors.append(f'symlink: {name}')
        elif not path.is_file():
            errors.append(f'missing: {name}')
        elif path.stat().st_size != entry.get('bytes') or sha256(path) != entry.get('sha256'):
            errors.append(f'size/hash mismatch: {name}')
    ignored_parts = {'.git', '.venv', '__pycache__', '.matplotlib'}
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if set(relative.parts) & ignored_parts or relative.parts[:2] == ('code', '附件'):
            continue
        name = relative.as_posix()
        if path.is_symlink():
            if f'symlink: {name}' not in errors:
                errors.append(f'symlink: {name}')
        elif path.is_file() and name != 'PUBLIC_MANIFEST.json' and name not in expected:
            errors.append(f'unexpected: {name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = verify(args.root)
    if errors:
        print('\n'.join(errors))
        return 1
    print('PASS: all public-manifest files match SHA-256; no unexpected payload files.')
    print('Official inputs and local runtime caches are excluded; this is not an original-freeze check.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

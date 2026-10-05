"""
Build the 3D anatomy the page shows, from two open datasets of real anatomy.

    Brain and outer ear:  BodyParts3D 4.0, (c) The Database Center for Life
                          Science, licensed CC BY-SA 2.1 Japan.
    Ear canal, eardrum,   The OpenEar library (Sieber et al., Scientific Data
    ossicles, cochlea,    2019, doi:10.5281/zenodo.1473724), specimen ZETA,
    nerve:                licensed CC BY 4.0.

Only the needed mesh files are read out of the remote archives. The ear
specimen, scanned on its own, is placed in the BodyParts3D head: its canal
opening goes to the right outer ear, its nerve end to the side of the pons,
and it is rolled so the ossicles sit above the eardrum, as they do in life.

Writes public/models/hearing.glb and src/data/anatomy.json:

    python tools/build_anatomy.py
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
import trimesh
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / '.cache'
BP3D = 'https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/'
OPENEAR = 'https://zenodo.org/api/records/1473724/files/ZETA.zip/content'
SCALE = 0.05  # scene units per millimetre

# BodyParts3D concept names that are drawn as their own part.
NAMED = {
    'auditory_cortex': ['posterior part of right superior temporal gyrus'],
    'colliculus': ['right inferior colliculus', 'left inferior colliculus'],
    'geniculate': ['right medial geniculate body', 'left medial geniculate body'],
    'thalamus': ['right thalamus', 'left thalamus'],
    'cerebellum': ['cerebellum'],
    'brainstem': ['pons', 'medulla oblongata', 'midbrain'],
}
# Everything else that belongs to the cerebral cortex is drawn as one shell.
SHELL_WORDS = ('gyrus', 'lobule', 'occipital lobe', 'insula')
EAR_FILES = {
    'canal': ['10_External Auditory Canal.ply'],
    'eardrum': ['09_Tympanic Membrane.ply'],
    'bones': ['03_Malleus.ply', '04_Incus.ply', '05_Stapes.ply'],
    'cochlea': ['01_Scala Tympani.ply', '02_Scala Vestibuli.ply'],
    'nerve': ['08_Cochleovestibular Nerve.ply'],
}


def cached(name, fetch):
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch())
    return path


def remote_text(name):
    import urllib.request
    return cached(name, lambda: urllib.request.urlopen(BP3D + name, timeout=120).read()).read_text(encoding='utf-8')


def element_names():
    """Element file id -> its most specific concept name, over both BodyParts3D trees."""
    owners = {}
    for listing in ('partof_element_parts.txt', 'isa_element_parts.txt'):
        for row in csv.reader(io.StringIO(remote_text(listing)), delimiter='\t'):
            if len(row) >= 3 and row[0].startswith('FMA'):
                owners.setdefault(row[1], set()).add(row[2])
    names = {}
    for concept, elements in owners.items():
        for element in elements:
            if element not in names or len(elements) < names[element][0]:
                names[element] = (len(elements), concept)
    return {element: name for element, (_, name) in names.items()}, owners


def load_bp3d(elements):
    """Download and load BodyParts3D element meshes, in millimetres."""
    meshes = {}
    missing = [e for e in elements if not (CACHE / 'bp3d' / f'{e}.obj').exists()]
    for archive in ('partof_BP3D_4.0_obj_99.zip', 'isa_BP3D_4.0_obj_99.zip'):
        if not missing:
            break
        with RemoteZip(BP3D + archive) as remote:
            inside = {Path(n).stem: n for n in remote.namelist()}
            for element in list(missing):
                if element in inside:
                    cached(f'bp3d/{element}.obj', lambda: remote.read(inside[element]))
                    missing.remove(element)
    for element in elements:
        meshes[element] = trimesh.load(CACHE / 'bp3d' / f'{element}.obj', force='mesh', process=False)
    return meshes


def load_openear():
    wanted = sorted({f for files in EAR_FILES.values() for f in files})
    if any(not (CACHE / 'openear' / f).exists() for f in wanted):
        with RemoteZip(OPENEAR) as remote:
            for f in wanted:
                cached(f'openear/{f}', lambda: remote.read('07_3D_Models/' + f))
    return {part: trimesh.util.concatenate(
        [trimesh.load(CACHE / 'openear' / f, force='mesh', process=False) for f in files])
        for part, files in EAR_FILES.items()}


def far_end(mesh, away_from, share=0.1):
    """Centre of the part of a mesh farthest from a point."""
    distance = np.linalg.norm(mesh.vertices - away_from, axis=1)
    return mesh.vertices[distance >= np.quantile(distance, 1 - share)].mean(axis=0)


def rotation_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def rotation_about(axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def place_ear(ear, canal_opening, nerve_root):
    """Rigidly move (and uniformly scale) the ear specimen into the head. Returns the scale used."""
    lateral = far_end(ear['canal'], ear['nerve'].centroid)
    medial = far_end(ear['nerve'], ear['canal'].centroid)
    turn = rotation_between(medial - lateral, nerve_root - canal_opening)
    axis = (nerve_root - canal_opening) / np.linalg.norm(nerve_root - canal_opening)

    def across(v):  # the part of a vector at right angles to the ear's axis
        return v - np.dot(v, axis) * axis

    above = across(turn @ (ear['bones'].centroid - ear['eardrum'].centroid))
    up = across(np.array([0.0, 0.0, 1.0]))
    angle = np.arctan2(np.dot(np.cross(above, up), axis), np.dot(above, up))
    turn = rotation_about(axis, angle) @ turn
    scale = np.linalg.norm(nerve_root - canal_opening) / np.linalg.norm(medial - lateral)
    for mesh in ear.values():
        mesh.vertices = (scale * (mesh.vertices - lateral)) @ turn.T + canal_opening
    return scale


def main():
    names, owners = element_names()
    brain = sorted(owners['brain'])
    extra = [e for part in NAMED.values() for concept in part for e in owners.get(concept, [])]
    stg = [e for concept in ('anterior part of right superior temporal gyrus',
                             'anterior part of left superior temporal gyrus',
                             'posterior part of left superior temporal gyrus') for e in owners[concept]]
    outer = sorted(owners['external ear'])
    meshes = load_bp3d(sorted(set(brain + extra + stg + outer)))

    parts = {}
    used = set()
    for part, concepts in NAMED.items():
        elements = sorted({e for concept in concepts for e in owners.get(concept, [])})
        used.update(elements)
        parts[part] = trimesh.util.concatenate([meshes[e] for e in elements])
    shell = [e for e in sorted(set(brain + stg)) if e not in used
             and any(word in names[e] for word in SHELL_WORDS)]
    parts['cortex'] = trimesh.util.concatenate([meshes[e] for e in shell])

    both_ears = trimesh.util.concatenate([meshes[e] for e in outer])
    right = both_ears.vertices[:, 0] < 0           # BodyParts3D: the body's right is -x
    keep = right[both_ears.faces].all(axis=1)
    parts['outer_ear'] = trimesh.Trimesh(both_ears.vertices, both_ears.faces[keep], process=True)

    ear_vertices = parts['outer_ear'].vertices
    opening = ear_vertices.mean(axis=0)
    opening[0] = np.quantile(ear_vertices[:, 0], 0.9)          # the side against the head
    pons = trimesh.util.concatenate([meshes[e] for e in owners['pons']])
    side = pons.vertices[(pons.vertices[:, 0] < 0) & (pons.vertices[:, 2] < pons.centroid[2])]
    nerve_root = side[side[:, 0] <= np.quantile(side[:, 0], 0.03)].mean(axis=0)

    ear = load_openear()
    scale = place_ear(ear, opening, nerve_root)
    parts.update(ear)

    centre = (parts['cortex'].bounds.mean(axis=0) + parts['outer_ear'].bounds.mean(axis=0)) / 2

    def to_scene(points):  # millimetres, +z up, -y forward  ->  scene units, +y up, +z toward the viewer
        p = (np.asarray(points, dtype=float) - centre) * SCALE
        return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)

    scene = trimesh.Scene()
    for name, mesh in parts.items():
        mesh = trimesh.Trimesh(to_scene(mesh.vertices), mesh.faces, process=True)
        mesh.fix_normals()
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    target = ROOT / 'public' / 'models' / 'hearing.glb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(scene.export(file_type='glb'))

    def spot(mesh):
        size = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))) * SCALE
        return {'position': np.round(to_scene(mesh.centroid), 3).tolist(), 'size': round(size, 3)}

    air = opening + np.array([-70.0, 0.0, 0.0])
    stops = {
        'sound': {'position': np.round(to_scene(air), 3).tolist(), 'size': 4.0},
        'outer-ear': spot(parts['outer_ear']),
        'eardrum': spot(parts['eardrum']),
        'bones': spot(parts['bones']),
        'cochlea': spot(parts['cochlea']),
        'hair-cells': spot(parts['cochlea']),
        'nerve': spot(parts['nerve']),
        'brainstem': {'position': np.round(to_scene(nerve_root), 3).tolist(), 'size': 2.4},
        'midbrain': spot(meshes[sorted(owners['right inferior colliculus'])[0]]),
        'thalamus': spot(meshes[sorted(owners['right medial geniculate body'])[0]]),
        'cortex': spot(parts['auditory_cortex']),
    }
    for stop in ('midbrain', 'thalamus'):
        stops[stop]['size'] = max(stops[stop]['size'], 1.6)
    anatomy = {'stops': stops, 'ear_scale': round(float(scale), 3),
               'faces': {name: int(len(mesh.faces)) for name, mesh in parts.items()}}
    (ROOT / 'src' / 'data' / 'anatomy.json').write_text(json.dumps(anatomy, indent=1) + '\n', encoding='utf-8')
    print(f"{target}: {target.stat().st_size / 1e6:.2f} MB, {sum(anatomy['faces'].values())} faces")
    print(f"ear specimen scaled by {scale:.3f} to fit the head; {len(shell)} cortex regions")
    for name, stop in stops.items():
        print(f"  {name:11s} {stop}")


if __name__ == '__main__':
    main()

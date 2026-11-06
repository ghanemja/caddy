"""Example scans and the scan → CAD session factory.

The examples stand in for a 3D scanner: a triangle mesh in an arbitrary pose with 0.01 mm noise. One is the
reference nipple tessellated by the kernel (so the right answer is known); the other is a flanged bushing meshed by
plain numpy (no CAD kernel involved), so nothing about it is known to the pipeline in advance. Any closed STL of a
turned part works the same way (`from_stl`).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import smlibpy as sm

from . import fit as F
from .check import Scan
from .session import ScanSession

HERE = Path(__file__).resolve().parents[2]

EXAMPLES = {
    "scan_nipple": dict(title="Scan → CAD: hex nipple", material={"name": "brass CW617N", "density_g_cm3": 8.47}),
    "scan_bushing": dict(title="Scan → CAD: flanged bushing", material={"name": "bronze CuSn8", "density_g_cm3": 8.8}),
}


def _pose(V: np.ndarray, seed: int) -> np.ndarray:
    """An arbitrary scanner pose and 0.01 mm noise."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=3); q /= np.linalg.norm(q)
    a = rng.uniform(0.4, 1.2)
    K = np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])
    R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return V @ R.T + rng.uniform(-20, 20, size=3) + rng.normal(scale=0.01, size=V.shape)


def revolve_mesh(profile: list[tuple[float, float]], n: int = 144) -> tuple[np.ndarray, np.ndarray]:
    """A closed (r, z) profile turned about z into a triangle mesh (numpy only)."""
    P = np.array(profile, float)
    m = len(P)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    V = np.array([[r * math.cos(t), r * math.sin(t), z] for (r, z) in P for t in th])
    T = []
    for i in range(m):
        j = (i + 1) % m
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            c, d = j * n + k, j * n + (k + 1) % n
            T += [(a, c, b), (b, c, d)]
    T = np.array(T)
    tri = V[T]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    if vol < 0:
        T = T[:, [0, 2, 1]]
    return V, T


BUSHING = [(6.0, 0.0), (13.0, 0.0), (14.0, 1.0), (14.0, 12.0), (22.0, 12.0), (22.0, 17.0),
           (12.0, 17.0), (10.0, 23.0), (9.0, 23.0), (9.0, 19.0), (6.0, 19.0)]


@lru_cache(maxsize=4)
def example_mesh(name: str) -> sm.Mesh:
    if name == "scan_bushing":
        V, T = revolve_mesh(BUSHING)
        return sm.Mesh(_pose(V, 7), T)
    if name == "scan_nipple":
        from caddy.agent.session import Session
        from caddy.kernel.build import tessellate
        s = Session()
        s.build_cad((HERE / "data" / "examples" / "nipple.sq.py").read_text())
        m = tessellate(s.reports[5].result, chord=0.01, angle=4)
        return sm.Mesh(_pose(m["points"].copy(), 1), m["tris"])
    raise KeyError(name)


def from_stl(data: bytes) -> sm.Mesh:
    return sm.from_stl(data)


@lru_cache(maxsize=4)
def measured(name: str):
    """The measuring steps for an example scan (cached: the scan doesn't change)."""
    return F.run(example_mesh(name))


def new_session(name: str, mesh: sm.Mesh | None = None, tol: float = 0.1) -> tuple[ScanSession, list[dict], str]:
    """Measure the scan (every step on SMLib), and open a session judged against it. Returns (session, steps,
    draft script)."""
    framed, steps, f, code = F.run(mesh) if mesh is not None else measured(name)
    ex = EXAMPLES.get(name, dict(title="Scan → CAD", material=None))
    s = ScanSession(Scan(name, framed), title=ex["title"], tol=tol, material=ex["material"])
    s.scan_steps, s.scan_fit, s.scan_draft = steps, f, code
    return s, steps, code

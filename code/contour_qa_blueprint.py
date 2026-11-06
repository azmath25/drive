"""Research prototype: Willmore/anisotropy and visible-silhouette QA for triangle meshes.

NOT a formal or certified mesh-quality validator. Curvatures are discretized and
must be calibrated through mesh refinement against analytic reference surfaces.

Dependencies: numpy, scipy, scikit-image, open3d (only for rendering functions).
Usage: python contour_qa_blueprint.py model.ply --views 64 --resolution 512
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull
from skimage.measure import find_contours


def curvature_topology(vertices, faces):
    """Vertex-area curvature approximation: cotangent H and angle-defect K.

    Assumes a correctly embedded, closed, manifold triangulation. No internal
    mesh repairs are performed. Returns diagnostics to guard against misuse.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    n = len(V)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError('Expected vertices (N,3) and triangular faces (M,3)')
    if not len(F) or np.any(F < 0) or np.any(F >= n):
        raise ValueError('Empty or invalid face indices')
    if len(np.unique(F)) != n:
        raise ValueError('Remove unreferenced vertices before checking topology')
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    twice_area = np.linalg.norm(cr, axis=1)
    if np.any(twice_area < 1e-14):
        raise ValueError('Degenerate triangles; clean mesh first')
    face_area = 0.5 * twice_area
    Av = np.bincount(F.ravel(), weights=np.repeat(face_area / 3, 3), minlength=n)

    # Angles at a, b, c. The denominator is twice the triangle area.
    aa = np.arctan2(twice_area, np.einsum('ij,ij->i', b-a, c-a))
    ab = np.arctan2(twice_area, np.einsum('ij,ij->i', c-b, a-b))
    ac = np.arctan2(twice_area, np.einsum('ij,ij->i', a-c, b-c))
    angle_sum = np.bincount(F.ravel(), weights=np.column_stack((aa, ab, ac)).ravel(), minlength=n)

    # Cotangents at corners; weighted opposite edges.
    cot_a = np.einsum('ij,ij->i', b-a, c-a) / twice_area
    cot_b = np.einsum('ij,ij->i', c-b, a-b) / twice_area
    cot_c = np.einsum('ij,ij->i', a-c, b-c) / twice_area
    ei = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    ej = np.concatenate((F[:, 2], F[:, 0], F[:, 1]))
    w = 0.5 * np.concatenate((cot_a, cot_b, cot_c))
    rows = np.concatenate((ei, ej))
    cols = np.concatenate((ej, ei))
    offdiag = sparse.coo_matrix((-np.concatenate((w, w)), (rows, cols)), shape=(n, n)).tocsr()
    stiffness = offdiag + sparse.diags(-np.asarray(offdiag.sum(axis=1)).ravel())
    H_vec = (stiffness @ V) / (2.0 * Av[:, None])
    H2 = np.einsum('ij,ij->i', H_vec, H_vec)
    K = (2.0 * np.pi - angle_sum) / Av

    edges = np.sort(np.concatenate((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]])), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(edge_counts != 2):
        raise ValueError('Mesh has boundary or nonmanifold edges; theorem not applicable')
    E = len(edge_counts)
    chi = int(n - E + len(F))

    # H^2-K is nonnegative for a smooth surface; check how badly the
    # independent discretizations depart from this compatibility condition.
    discriminant = H2 - K
    scale = H2 + np.abs(K) + 1e-12
    unphysical = discriminant < -0.05 * scale
    unphysical_fraction = float(np.sum(Av[unphysical]) / np.sum(Av))

    # Recover absolute principal curvatures up to an overall normal sign.
    # Clipping slightly negative discriminants is ONLY a numerical heuristic.
    Habs = np.sqrt(np.maximum(H2, 0.0))
    deviation = np.sqrt(np.maximum(discriminant, 0.0))
    k1 = Habs + deviation
    k2 = Habs - deviation
    q = k1*k1 + k2*k2
    numerator = (k1*k1 - k2*k2)**2
    d_density = np.divide(numerator, q, out=np.zeros_like(q), where=q > 1e-20)
    D = float(np.dot(Av, d_density))
    W = float(np.dot(Av, H2))
    A = float(face_area.sum())
    # Discrete angle defects satisfy Gauss-Bonnet exactly up to roundoff.
    gb_residual = float(np.dot(Av, K) - 2 * np.pi * chi)
    return dict(A=A, W=W, D=D, chi=chi, k1=k1, k2=k2,
                H2=H2, K=K, vertex_area=Av,
                gauss_bonnet_residual=gb_residual,
                unphysical_curvature_area_fraction=unphysical_fraction,
                n_vertices=n, n_faces=len(F))


def fibonacci_views(n):
    i = np.arange(n, dtype=np.float64)
    z = 1 - 2 * (i + 0.5) / n
    ang = 2 * np.pi * i / ((1 + np.sqrt(5)) / 2)
    r = np.sqrt(np.maximum(0, 1 - z*z))
    return np.column_stack((r*np.cos(ang), r*np.sin(ang), z))


def camera_axes(view):
    view = np.asarray(view, dtype=np.float64)
    view = view / np.linalg.norm(view)
    helper = np.array([0., 0., 1.]) if abs(view[2]) < 0.9 else np.array([0., 1., 0.])
    ex = np.cross(view, helper); ex /= np.linalg.norm(ex)
    ey = np.cross(view, ex)
    return view, ex, ey


def build_scene(vertices, faces):
    """Open3D is optional for curvature-only use; required for raycasting."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32))
    )
    return scene


def orthographic_mask(scene, vertices, view, resolution=512, margin=1.2):
    """First-hit rays: all visible foreground pixels, including visible holes."""
    import open3d as o3d
    V = np.asarray(vertices, dtype=np.float64)
    center = 0.5 * (V.min(axis=0) + V.max(axis=0))
    radius = float(np.linalg.norm(V - center, axis=1).max())
    view, ex, ey = camera_axes(view)
    halfwidth = max(radius * margin, 1e-8)
    xs = (np.arange(resolution) + 0.5 - resolution / 2.0) * (2*halfwidth / resolution)
    xx, yy = np.meshgrid(xs, xs)
    origins = center - 3*radius*view + xx[..., None]*ex + yy[..., None]*ey
    rays = np.concatenate((origins, np.broadcast_to(view, origins.shape)), axis=2).astype(np.float32)
    hits = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    return np.isfinite(hits), 2*halfwidth/resolution


def mask_perimeter(mask, meters_per_pixel):
    """Marching squares estimate; validate pixel-size and raster bias."""
    contours = find_contours(np.pad(mask.astype(float), 1), 0.5)
    return float(meters_per_pixel * sum(
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum() for path in contours
    ))


def mean_visible_perimeter(vertices, faces, views=64, resolution=512):
    scene = build_scene(vertices, faces)
    vals = []
    for u in fibonacci_views(views):
        mask, pitch = orthographic_mask(scene, vertices, u, resolution)
        vals.append(mask_perimeter(mask, pitch))
    return float(np.mean(vals)), np.asarray(vals)


def mean_hull_perimeter(vertices, views=64):
    """Mean perimeter of planar convex hulls; independent comparison."""
    V = np.asarray(vertices, dtype=np.float64)
    vals = []
    for u in fibonacci_views(views):
        _, ex, ey = camera_axes(u)
        planar = np.column_stack((V@ex, V@ey))
        vals.append(ConvexHull(planar).area)  # 2-D ConvexHull.area IS perimeter
    return float(np.mean(vals))


def certificate(curvature, observed_mean_s, numerical_margin=0.0):
    """Positive G means apparent violation; not automatically wrong geometry.

    numerical_margin is a dimensionless tolerance normalized by W, selected
    and frozen using calibration meshes; it must NOT be hand-tuned on test data.
    """
    A, W, D, chi = (curvature[k] for k in ('A', 'W', 'D', 'chi'))
    if A <= 0 or W <= 0:
        raise ValueError('Nonpositive area/Willmore estimate')
    G = float(np.pi*chi + 2*observed_mean_s**2 / A + D/32 - W)
    z = G / W
    # Unreliable curvature must yield an abstention rather than a rejection.
    if curvature['unphysical_curvature_area_fraction'] > 0.01:
        decision = 'ABSTAIN_UNRELIABLE_CURVATURE'
    else:
        decision = 'FLAG_CONSTRAINT_VIOLATION' if z > numerical_margin else 'NO_VIOLATION_DETECTED'
    return dict(G=G, normalized_gap=z, decision=decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh_path')
    parser.add_argument('--views', type=int, default=64)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--margin', type=float, default=0.02)
    args = parser.parse_args()
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    if mesh.is_empty():
        raise ValueError('Empty mesh')
    if not (mesh.is_edge_manifold(allow_boundary_edges=False)
            and mesh.is_vertex_manifold()
            and mesh.is_orientable()
            and not mesh.is_self_intersecting()):
        raise ValueError('Nonclosed, nonmanifold, nonorientable or self-intersecting mesh')
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    curv = curvature_topology(V, F)
    s, view_samples = mean_visible_perimeter(V, F, args.views, args.resolution)
    hull = mean_hull_perimeter(V, args.views)
    cert = certificate(curv, s, args.margin)
    summary = {key: val for key, val in curv.items()
               if key in ('A','W','D','chi','gauss_bonnet_residual',
                          'unphysical_curvature_area_fraction','n_vertices','n_faces')}
    summary.update(s_visible=s, s_hull=hull, **cert)
    print(json.dumps(summary, indent=2))
    print('NOTICE: same-mesh certificates cannot establish ground-truth reconstruction quality.')


if __name__ == '__main__':
    main()
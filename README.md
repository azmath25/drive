# 3D Contour-Generator and Willmore–Anisotropy Research Archive

Original mathematics, AI-assisted proof audits, novelty evaluation, 3D-reconstruction/QA experimental blueprint, and Python research prototypes consolidated from a research conversation (November 2026).

**This repository preserves source artifacts. It is not a peer-reviewed paper, certified geometry validator, or fully tested reconstruction pipeline.** Some files discuss hypotheses, corrected statements, unverified novelty, and numerical experiments. Read [Research Status](docs/RESEARCH_STATUS.md) before using any inequality as a rejection test.

## Quick start for a collaborator

If you only want the PDFs, open the [`papers/`](papers/) directory in GitHub and select a report. Use the page's **Download raw file** icon, or choose **Code → Download ZIP** on the repository home page to obtain all files. The files are hosted in GitHub, so a collaborator does not need access to the original ChatGPT conversation.

If you want to run the code:

```bash
# Clone using HTTPS (works without an SSH key for this public repo)
git clone https://github.com/azmath25/drive.git
cd drive

# Python 3.10/3.11 recommended for Open3D wheel compatibility
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tests/test_analytic_smoke.py
```

For a minimal CPU-only mathematical environment, install `numpy` and `scipy` instead of the full requirements set. Open3D is needed **only** for the mesh raycasting functions; the sphere/torus numerical test uses NumPy and SciPy.

## File map

| Path | Contents | Use |
| --- | --- | --- |
| [`papers/original/`](papers/original/) | Exact original two-page PDF | Original statement and attribution |
| [`papers/revised/`](papers/revised/) | Six-page revised PDF, editable `.tex`; verification report PDF and `.tex` | Proof reconstruction and caveats |
| [`papers/assessments/`](papers/assessments/) | Novelty report, mathematical reconstruction audit, 6-page QA experimental blueprint PDF and `.md` | Literature context and evaluation plan |
| [`code/numerical_verification.py`](code/numerical_verification.py) | Independent sphere/torus geometry checks | CPU numerical consistency tests |
| [`code/contour_qa_blueprint.py`](code/contour_qa_blueprint.py) | Curvature, topology, silhouette, convex-hull and residual prototype | Mesh-level research experimentation |
| [`code/numerical_results.txt`](code/numerical_results.txt) | Prior captured numerical output | Historical reference, not CI proof |
| [`reference_images/`](reference_images/) | Two supplied screenshots containing proposed improvements | Research discussion history |
| [`archives/`](archives/) | Exact original downloadable ZIP packages | Preservation/backup of generated deliverables |
| [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) | Claim ledger and validity caveats | **Read before research use** |

## Mathematical quantity definitions

Let `S` be a smooth compact closed oriented embedded surface; `A` its area; `chi` its Euler characteristic; `W = integral H^2 dA`, where `H = (k1+k2)/2`. Define `D = integral ((k1^2-k2^2)^2/(k1^2+k2^2)) dA`, with zero convention when both curvatures vanish.

The refined estimate established in the audit is

```text
L_rim^2 <= (A/2) * (W - pi*chi) - (A/64) * D
```

`L_rim` is the **direction-average spatial length of the entire 3D contour generator**. In the observation-conditioned QA setting, `s_visible` denotes the uniformly averaged **visible projected** silhouette perimeter, giving the necessary condition

```text
G = pi*chi + 2*s_visible^2/A + D/32 - W <= 0
```

These are distinct quantities. In particular, a projected convex-hull perimeter is not interchangeable with the full 3D rim. The known inequality with `A*W/4` is **not valid** for the spatial rim in general.

## Running the geometry tests

```bash
# Fast CPU-only sanity checks
python tests/test_analytic_smoke.py

# Detailed sphere/torus comparisons using two independent integrations:
# computationally heavier (can take several minutes depending on CPU)
python code/numerical_verification.py
```

These tests check analytical examples and quadrature, **not reconstruction accuracy**. They require no data download, cloud service, GPU or paid APIs.

## Evaluating a candidate mesh

```bash
python code/contour_qa_blueprint.py /path/to/your/closed_mesh.ply \
  --views 64 --resolution 512 --margin 0.02
```

Supported input depends on Open3D; examples include `.ply`, `.obj`, `.stl`. The script checks whether the mesh is nonempty, edge/vertex manifold, orientable and non-self-intersecting, then approximates `A, chi, W, D`, averaged visible silhouette length and averaged projected convex-hull perimeter. It prints a JSON summary and a decision.

- `NO_VIOLATION_DETECTED`: numerical consistency only; **does not imply geometric correctness**.
- `FLAG_CONSTRAINT_VIOLATION`: apparent violation exceeding a user-supplied threshold; inspect estimator/raster/scale errors before rejecting.
- `ABSTAIN_UNRELIABLE_CURVATURE`: the estimated mean and Gaussian curvature are incompatible over too much area.

**Important:** The default command computes `s_visible` from the *same* candidate mesh and cannot by itself filter out geometrically wrong, but valid, reconstructions. To test real QA, use `certificate(geometry, observed_mean_s=...)` with an *independent*, metrically calibrated, sufficiently sampled set of observed silhouettes. This is a research API, not a turnkey multi-camera calibration loader. No empirically validated decision threshold ships with the repository.

Example (Python):

```python
import numpy as np
from code.contour_qa_blueprint import curvature_topology, certificate

# vertices shape (N,3); faces shape (M,3), from a valid closed triangle mesh
geometry = curvature_topology(vertices, faces)
# s_obs must be independently measured, in the same physical units
result = certificate(geometry, observed_mean_s=s_obs, numerical_margin=0.02)
print(result)
```

If `code` is not importable as a namespace package in your environment, use `PYTHONPATH=.` or `sys.path.insert(0, 'code')` and import `contour_qa_blueprint` instead.

## Experimental blueprint (A/B reconstruction and QA)

Read [`papers/assessments/Contour_QA_Reconstruction_Experimental_Blueprint.md`](papers/assessments/Contour_QA_Reconstruction_Experimental_Blueprint.md) alongside the PDF. Recommended arms:

1. Standard multi-view reconstruction + direct silhouette matching.
2. Add conventional Willmore regularization.
3. Add the self-consistent bound on predicted silhouettes (negative control).
4. Add an observation-conditioned silhouette/Willmore penalty without `D`.
5. Add the same penalty with `D/32`.

Use identical initializations, view splits, camera calibration and computational budgets; evaluate against independently held-out 3D geometry. Report Chamfer distance, F-score, silhouette IoU, normal consistency, topology correctness, runtime and calibrated **false rejection/false acceptance** at fixed false-positive rates. Include sphere, ellipsoid, capsule, rounded cylinder, torus, multi-handle, open/closed cavity and mesh-resolution stress tests. An enclosed invisible cavity cannot be recovered from exterior silhouettes alone.

The proposed penalty can be minimized by **increasing curvature energy** and possibly producing artificial wrinkles. Monitor this failure mode rather than assuming a lower penalty implies a better surface.

## Rebuilding editable manuscripts

On a computer with XeLaTeX and the font packages required by the sources:

```bash
cd papers/revised
xelatex contour_generator_revised.tex
xelatex contour_generator_revised.tex
xelatex verification_report.tex
xelatex verification_report.tex
```

The original compiled PDFs are already present; no TeX installation is required to read them. Rebuilding might require fonts or TeX packages not installed on your computer; do not assume pixel-identical output.

## Notes for reviewers and collaborators

- Do **not** mistake a satisfied inequality for accurate 3D reconstruction.
- Uniform directional averages cannot be inferred exactly from a small nonuniform set of cameras without extra modeling assumptions.
- Ensure physical scale: `A` is squared length, `s` is length, `W/D/chi` are dimensionless.
- Do not apply smooth closed-surface theorems blindly to open, self-intersecting, nonmanifold or low-resolution meshes.
- Numerical curvature estimates can falsely reject spheres close to equality; calibration and abstention are mandatory.
- Older documents reflect historical hypotheses and may contain claims corrected later. The original sources and both archive ZIPs are retained deliberately.
- Source mathematics attributed to **İmamverdi Bədəlov**; the extended manuscripts are AI-assisted audits. Human review of authorship and bibliographic priority is essential before publication.

## Repo sharing and use

Use the stable URL: **https://github.com/azmath25/drive**. Others can read and download the research archive without a ChatGPT account or sandbox permissions. For an individual PDF, open it under `papers/` on GitHub and choose its download control. For all research files at once, download the repository ZIP from GitHub's **Code** menu.

No credentials, production services, or cloud deployment are needed. No license has been granted for unrestricted redistribution or modification of authored research; coordinate with the original author for publication or commercial use.

---
title: "Testing a Willmore-Anisotropy Inequality for 3D Reconstruction and Mesh QA"
subtitle: "Pre-registration-ready experimental blueprint and implementation audit"
author: "Independent mathematical and computer-vision assessment"
date: "6 November 2026"
geometry: margin=0.75in
fontsize: 10pt
mainfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: blue
---

# Executive verdict

**Scientific status:** The proposed strengthened inequality is mathematically meaningful as a bound on **spatial rim length**. For *uniformly averaged visible orthographic silhouettes* it is also a valid, weaker necessary bound. But a self-consistency penalty evaluated on the **same candidate mesh** is identically inactive in the continuum on all admissible meshes, including incorrect reconstructions. Consequently, it cannot inherently be a discriminator of reconstruction quality.

**Primary falsifiable question:** Does combining externally observed uniformly sampled silhouette perimeter with the curvature terms of a candidate mesh improve reconstruction or reject inconsistent reconstructions beyond a properly tuned direct silhouette loss and standard Willmore regularization? This is empirical, unproved, and vulnerable to noisy curvature estimation.

# 1. Definitions and mathematical validity

Let $S\subset\mathbb R^3$ be a smooth, compact, oriented, embedded, boundaryless surface; $A=\int_SdA$, $W=\int_S H^2 dA$ with $H=(\kappa_1+\kappa_2)/2$, and $\chi$ its Euler characteristic. Define

$$D=\int_S\frac{(\kappa_1^2-\kappa_2^2)^2}{\kappa_1^2+\kappa_2^2}\,dA,$$

with zero integrand at zero-curvature points. If $L_{\mathrm{rim}}$ is the directional mean of **full spatial rim length**, the audited refinement reads

$$L_{\mathrm{rim}}^2\le\frac A2(W-\pi\chi)-\frac A{64}D.$$

For a visible orthographic silhouette of an opaque object, let $s_{\rm vis}$ be the uniformly averaged planar perimeter, counting image-set boundary components consistently. Since $s_{\rm vis}\le L_{\rm rim}$, one obtains the necessary condition

$$\boxed{W\ge\pi\chi+\frac{2s_{\rm vis}^2}{A}+\frac D{32}.}\tag{1}$$

For **every** smooth admissible surface, equation (1) holds when the measurements come from that same surface. A candidate-only hinge $[\pi\chi+2s_{\rm pred}^2/A+D/32-W]_+$ therefore supplies no theoretical gradient for optimization. This is a **structural null result**, not an implementation flaw.

For a candidate $\hat S$ and independently observed $s_{\rm obs}$, the residual

$$G(\hat S,s_{\rm obs})=\pi\hat\chi+\frac{2s_{\rm obs}^2}{\hat A}+\frac{\hat D}{32}-\hat W$$

can be positive. It then signals an incompatibility of this candidate's intrinsic descriptors with the external silhouette summary **assuming comparable metric calibration, appropriate viewing directions, and reliable estimates**. It does not directly prove the candidate is geometrically wrong.

**Failure modes and limits.** (i) $W$ can grow due to small-scale wrinkles, so false shapes can easily satisfy the condition; (ii) closed invisible internal cavities cannot be recovered from silhouettes; (iii) genus/chi is unstable for noisy topology; (iv) real cameras typically have limited, nonuniform viewing directions; (v) $s^2/A$ and $W,D,\chi$ are invariant under uniform scaling, so the inequality cannot recover absolute scale; (vi) open surfaces, triangle soups, and self-intersecting meshes lie outside the theorem.

**Comparison caveat.** A convex-hull width/perimeter inequality may be evaluated for connected nonconvex surfaces, but the *hull* discards cavities and concavities. It is not equivalent to the original full-rim inequality.

# 2. Hypotheses, arms, and fair A/B evaluation

Use exactly the same input images, camera calibrations, initialization, backbone architecture, training steps, image/silhouette losses, mesh resolution, and random seeds for paired runs.

**Null hypotheses**: H0-R (reconstruction): adding the new penalty does not improve held-out 3D accuracy compared with direct silhouette training. H0-Q (QA): the observation-conditioned residual does not increase detection at a fixed false-rejection rate compared with simple silhouette disagreement, mesh validity checks, and conventional smoothness metrics.

| Arm | Losses / score | Purpose |
|---|---|---|
| R0 | Photometric/depth loss + direct per-view silhouette loss + standard mesh priors | Primary baseline |
| R1 | R0 + tuned conventional Willmore/bending regularizer | Tests regular curvature regularization |
| R2 | R0 + candidate-only hinge using its own predicted $s$ | Negative control; theoretically always zero |
| R3 | R0 + externally observed $s_{\rm obs}$ with $D=0$ | Tests observation-conditioned unrefined inequality |
| R4 | R0 + externally observed $s_{\rm obs}$ with full $D/32$ | Tests incremental value of anisotropy |

Optional R5: use an independently computed convex-hull mean-width condition as a separate QA baseline, not a replacement for the same-mesh or observation-conditioned arms.

For R4 use an explicitly named experimental loss (it is **not** a standard shape prior):

$$\mathcal L_{\rm extra}=\lambda\left[\max\left(0,\frac{\pi\chi+2s_{\rm obs}^2/A+D/32-W}{W+\epsilon}\right)\right]^2.$$

This term can perversely reward **increasing** $W$ by adding wrinkles, changing area, or reducing anisotropy; inspect each route. Evaluate an additional gradient-stopped $W$ variant to isolate optimization effects. For fixed-topology mesh deformation $\chi$ is constant; for neural SDFs, topology changes during remeshing, so apply discrete evaluations at checkpoints rather than assuming differentiability.

Do not backpropagate through Open3D raycasting: it is an evaluation renderer. Train silhouettes with a differentiable renderer such as PyTorch3D, keeping Open3D as an independent test oracle.

**Design size.** Pilot: 48 shapes (balanced by family), two camera regimes, three paired seeds, arms R0/R1/R3/R4: 1,152 runs. Add R2 to a smaller numerical diagnostic subset. Confirmatory test: approximately 240 held-out shapes, two camera regimes, three paired seeds, four selected arms: 5,760 runs. Report wall time and GPU-hour cost. If budget is tight, reduce shape count via paired power calculations from pilot variance, not via cherry-picking scenes.

**Cameras:** 4 and 12 training views for sparse and moderate regimes; 32+ held-out, approximately uniform orthographic test views. Repeat separately with perspective cameras to quantify theorem/model mismatch. Rotate view sets across seeds. A limited 180-degree real-camera trajectory is **not** an unbiased spherical mean.

# 3. Dataset and corruption factorial design

| Family | Diagnostic role | Recommended generation |
|---|---|---|
| Spheres, ellipsoids | Equality, umbilicity, anisotropy, near-convexity | Analytic parametric surfaces |
| Rounded long cylinders, capsules | Sharp convex regime; curvature imbalance | Smooth implicit/CAD constructions |
| Ring tori, genus 1 | Saddle regions, silhouette holes and occlusion | Parametric torus with $R/r$ sweep |
| Genus 2/3+ handlebodies | Topology and narrow necks | Smooth implicit level sets, manifold meshing |
| Concave bowls and star-shaped indents | Hidden/nonconvex contours | Controlled smooth deformations |
| Sealed internal voids | Fundamental silhouette non-identifiability | Shells with interior boundary components |

Generate references analytically whenever possible. Split by **source shape**, not by mesh tessellation or rendered image: all remeshings/views of a shape belong to the same split. Hold out entire shape families for OOD evaluation. Include clean/high-quality meshes for theorem validation, and intentionally invalid meshes only in the **separate invalid-input gate**.

**Public datasets:** ABC supplies CAD surface descriptions and geometric ground truth for curvature validation; DTU and Tanks and Temples supply multi-view reconstruction settings; Thingi10K deliberately includes manifold defects and should be labeled as invalid/input-QA stress data rather than naively assumed to satisfy the theorem. Sources: [ABC](https://openaccess.thecvf.com/content_CVPR_2019/html/Koch_ABC_A_Big_CAD_Model_Dataset_for_Geometric_Deep_Learning_CVPR_2019_paper.html), [Tanks and Temples](https://www.tanksandtemples.org/download/), [Thingi10K](https://github.com/Thingi10K/Thingi10K).

**Perturbation grid.** At three intensities each: isotropic/anisotropic remeshing, controlled high-frequency wrinkles, vertex noise, spike injection, over-smoothing, concavity filling, adding/removing handles, sealed invisible cavities, silhouette misalignment/calibration error, nonmanifold edges, holes in triangulation, self-intersections. A 'geometrically incorrect but valid' mesh must be distinguished from a 'mesh invalid for the theorem'.

**Hidden cavities:** In a sealed void, two objects can have identical exterior silhouettes while different geometry, topology, area, and curvature. Measure whether a method *abstains* rather than falsely claiming it has discovered the void. Other modalities (depth, tomography, multiview interior access) are required to resolve it.

# 4. Metrics, labels, and statistical analysis

**Reconstruction primary outcome:** mesh-to-reference symmetric surface-sampled Chamfer distance, normalized by object bounding-box diagonal. Secondary: normal consistency, F-score at preregistered geometric tolerances, topological match ($\chi$, connected components, cavity count), watertightness, held-out silhouette IoU/boundary-distance, and runtime/VRAM. Report curvature-energy drift and high-frequency roughness separately to expose gaming the constraint.

**QA labels:** define ACCEPTABLE using independent 3D ground truth (e.g. preregistered normalised Chamfer and F-score thresholds, no unacceptable topology error); define UNACCEPTABLE using those same independent criteria, not inequality satisfaction. For genuine real-world scans where 3D truth is unavailable, use blind expert ratings as secondary labels only, with inter-rater agreement.

Let *positive* mean an unacceptable candidate that should be rejected. Then

$$\mathrm{FPR}=\frac{\mathrm{acceptable\ rejected}}{\mathrm{all\ acceptable}},\qquad\mathrm{FNR}=\frac{\mathrm{unacceptable\ accepted}}{\mathrm{all\ unacceptable}}.$$

Measure AUROC, AUPRC, precision, recall, balanced accuracy, risk-coverage curves with abstention, calibration, and rejection recall **at exactly 1% and 5% FPR**. Thresholds are tuned on validation data then frozen; one does not tune them on held-out tests. Compare to (a) held-out silhouette IoU/mask mismatch, (b) conventional Willmore energy, (c) mesh validity tests, (d) hull-perimeter checks, (e) composite discriminators with and without the new normalized residual.

**Inference:** use paired differences by *shape*, stratify by shape family and view regime, and cluster bootstrap by source object rather than treating thousands of renderings as independent samples. Report 95% confidence intervals, familywise/multiple-comparison correction, and effect sizes, not just significance. Recommended advancement gate: statistically credible improvement in 3D Chamfer or QA recall at fixed FPR, no unacceptable rise in roughness, and a reasonable runtime cost. This is a prespecified *decision rule*, not an asserted result.

# 5. Numerical and software QA plan

1. **Validity gate:** Reject theorem evaluation on empty, degenerate, open, nonmanifold, nonorientable, or self-intersecting meshes; flag numerical triangles and extreme aspect ratios. Euler characteristic $V-E+F$ is only meaningful after appropriate mesh validation.
2. **Curvature:** cotangent Laplacian gives vector mean curvature; angle defects give Gaussian curvature. Alternatively, estimate $\kappa_1,\kappa_2$ through multi-scale tangent quadratic fits. Compare both and analytic CAD values.
3. **Anisotropy identity:** one may avoid explicitly resolving signs of principal curvatures by using
   $$\frac{(\kappa_1^2-\kappa_2^2)^2}{\kappa_1^2+\kappa_2^2}=\frac{16H^2(H^2-K)}{4H^2-2K},$$
   with a zero-curvature convention. Negative estimates of $H^2-K$ indicate discretization inconsistencies and require flags, not silent certification.
4. **Silhouettes:** uniformly sample $S^2$ with Fibonacci or scrambled Sobol views; cast first-hit *orthographic* rays into a calibrated grid; extract all visible boundary components from the binary foreground. Ray-tracing with Open3D uses `RaycastingScene.cast_rays`; the result uses infinite hit distance for misses. Camera pixel pitch must be converted to object-length units. Compare image raster resolutions (256/512/1024+) and camera rotations to quantify perimeter bias.
5. **Convergence:** icospheres and parametric tori at three or more mesh resolutions; independently increase view counts and image resolutions. Check scale invariance, rigid-motion invariance, Gauss-Bonnet residual, sphere equality, and torus analytic area/Willmore values.
6. **Uncertainty-aware decision:** bootstrap directions and repeat across rendering resolution and mesh refinements. Obtain a conservative interval $[G_{\rm low},G_{\rm high}]$ including systematic discretization uncertainty. Reject only when $G_{\rm low}>0$; pass inequality check only when $G_{\rm high}\le 0$; otherwise **ABSTAIN**. Even a conclusive violation means 'descriptors inconsistent', not 'surface geometrically bad'.

**Observed smoke test (local Python curvature-only check, no Open3D installed):** For unit icospheres subdivided 2/3/4 times, applying the *exact* sphere's $s=2\pi$ to discrete estimates produced apparent positive residuals $G=0.36346, 0.09230, 0.02321$. A zero-threshold QA gate falsely rejects all three. Estimated $W$ values converge $12.3452,12.5106,12.5524$ toward $4\pi=12.5664$. This is concrete evidence that numerical tolerance is essential.

The same implementation on tori with major/minor radii $(2,1)$ gave estimated $W=22.7092,22.7718,22.7876$ as resolution increased, tending to the analytic $4\pi^2/\sqrt3\approx22.7929$. The Python file is syntax-checked and curvature-tested. **The Open3D raycasting functions were not executed in this runtime; their API is grounded in Open3D documentation and must be smoke-tested on a machine with Open3D installed.**

# 6. Repository structure and usage

Recommended organization:

```text
experiments/
  configs/{r0,r1,r2,r3,r4}.yaml
  datasets/{shapes,views,splits}.json
  curvatures.py
  silhouette_renderer.py
  train.py
  qa_filter.py
  analyze.py
  tests/{sphere,torus,mesh_validity,raster_resolution}.py
```

The accompanying standalone file `contour_qa_blueprint.py` includes: cotangent and angle-defect curvature, Euler characteristic, approximate principal curvatures, anisotropy, spherical directions, Open3D orthographic raycasts, raster silhouette perimeter, projected convex-hull perimeter, normalized residual, and conservative abstention for incompatible curvature estimates.

To run after installing `numpy scipy scikit-image open3d`:

```bash
python contour_qa_blueprint.py candidate.ply --views 64 --resolution 512
```

It is a **research prototype**, not a certified numerical verifier or a differentiable training implementation.

# 7. Conditions for a positive research claim

The inequality will be **reconstruction-useful** only if R4 beats R0/R1 on independent held-out geometry and does not obtain its benefit from increasing $W$ via artificial wrinkles. It will be **QA-useful** only if the externally conditioned score improves rejection over direct observed-vs-rendered silhouette comparison at fixed 1%/5% false-rejection rates, remains calibrated on fine meshes and noisy masks, and correctly abstains on unverifiable inputs. If candidate-only R2 produces a large positive effect, inspect the implementation first: an exact theorem predicts no continuum signal, and such an effect likely comes from numerical artifacts or implicit added regularization.

**Overall verdict before experiments:** mathematically interesting; implementation QA plausible; statistically useful external-consistency filter unproven; inherent quality-aware candidate-only regularizer unlikely. Do not equate a tight inequality with a high-information reconstruction loss.

# Selected technical references

1. Open3D, [RaycastingScene API](https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html).
2. PyTorch3D, [SoftSilhouetteShader](https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/shader.html).
3. Koch et al., [ABC: A Big CAD Model Dataset for Geometric Deep Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Koch_ABC_A_Big_CAD_Model_Dataset_for_Geometric_Deep_Learning_CVPR_2019_paper.html), CVPR 2019.
4. [Tanks and Temples benchmark](https://www.tanksandtemples.org/download/).
5. Zhou and Jacobson, [Thingi10K](https://arxiv.org/abs/1605.04797), 2016.
6. Topping, [Sharp diameter estimates via integral geometry](https://arxiv.org/abs/2610.34367), October 2026.
7. Miura, [Topping's diameter conjecture for immersed surfaces](https://arxiv.org/abs/2611.01589), November 2026.
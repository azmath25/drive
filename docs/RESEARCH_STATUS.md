# Mathematical and experimental status

This repository preserves the supplied original theorem note, the successive AI-assisted review files, and the research-prototype code. **Preservation does not constitute a fresh independent peer review.**

## Definitions must not be conflated

- `L_rim`: mean **spatial** length of the entire contour generator, including hidden components. This is the main quantity in the original theorem.
- `L_projected`: mean length of projected contour-generator branches counted with multiplicity.
- `s_visible`: mean perimeter/length of the visible image silhouette, including visible holes under the selected definition.
- `L_hull`: mean perimeter of the 2D projection of the **3D convex hull**.

The original inequality is, for a smooth compact boundaryless oriented embedded surface,

`L_rim^2 <= (A/2) * (W - pi*chi)`.

The documented anisotropy refinement, with `D = integral (k1^2-k2^2)^2/(k1^2+k2^2) dA`, is

`L_rim^2 <= (A/2) * (W - pi*chi) - (A/64)*D`.

Since `s_visible <= L_rim` under the ordinary orthographic definition,

`W >= pi*chi + 2*s_visible^2/A + D/32`

is a **necessary condition**, not a sufficient accuracy test.

The proposed alternative `L_rim^2 <= A*W/4` is false; see the torus counterexample in the discussion and reports. A convex-hull perimeter bound with a similar right side concerns a *different measurement* and must not be substituted for `L_rim`.

## Critical caveat for QA

If all geometric quantities and silhouettes are computed consistently from the **same** valid smooth surface, the inequality is automatically satisfied (up to numerical errors), irrespective of the surface's ground-truth reconstruction accuracy. To test discriminatory usefulness, compare **independently observed** calibrated silhouettes to a reconstructed mesh and benchmark against stronger direct image losses.

The mesh implementation is heuristic: its cotangent-Laplacian and angle-defect curvature estimates need calibration; the rasterized silhouette depends on resolution, sampling and camera model. A small positive residual is **not** a mathematical counterexample.

## Novelty / attribution / reproducibility

- No claim is made here that any formula is bibliographically new.
- The original two-page note was supplied under the name İmamverdi Bədəlov.
- The six-page reconstruction and additional statements were produced in an AI-assisted audit; author permission and attribution must be resolved before publication.
- The numerical scripts do not constitute a formal proof or a tested end-to-end 3D reconstruction system.
- The manuscript's references and recent-priority claims should be checked manually before citation.
- Curvature and topology hypotheses (smoothness, closedness, embeddedness where needed, orientation) are essential.

## Appropriate research claims

Valid: a reproducible hypothesis-test blueprint and a non-certified QA consistency check.

Not established: reconstruction accuracy gains, calibrated FPR/FNR on any public benchmark, real-world deployment quality, or universal superiority to standard Willmore regularization.

"""Small CPU-only mathematical smoke tests; not a reconstruction benchmark."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from numerical_verification import ellipse_perimeter, test_sphere


def main():
    for radius in (0.5, 1.0, 3.0):
        L, bound = test_sphere(radius)
        assert abs(L - 2 * np.pi * radius) < 1e-12
        assert abs(L - bound) < 1e-11
    # Unit circle: perimeter 2*pi; collapse to segment: limiting perimeter 4.
    assert abs(float(ellipse_perimeter(1.0, 1.0)) - 2 * np.pi) < 1e-12
    assert abs(float(ellipse_perimeter(1.0, 0.0)) - 4.0) < 1e-12
    print("PASS: sphere and curvature-ellipse sanity checks")


if __name__ == "__main__":
    main()

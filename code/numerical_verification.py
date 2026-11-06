"""Deterministic numerical checks for the 3D contour-average identities.

Independent torus computations:
    (A) integrate curvature-ellipse formula over the surface;
    (B) parameterize the rim for each nonexceptional viewing direction,
        integrate its spatial length, then average directions;
    (C) likewise integrate its projected length and compare with the
        predicted normal-curvature integral.

This script is a check, not a substitute for the proofs in the manuscript.
"""
import numpy as np
from scipy.special import ellipe
from scipy.integrate import simpson


def ellipse_perimeter(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    big = np.maximum(np.abs(a), np.abs(b))
    small = np.minimum(np.abs(a), np.abs(b))
    return 4 * big * ellipe(1 - np.divide(small**2, big**2,
                                          out=np.zeros_like(big), where=big != 0))


def gauss_nodes(a,b,n):
    x,w=np.polynomial.legendre.leggauss(n)
    return (a+b)/2 + (b-a)/2*x, (b-a)/2*w


def curvature_average_torus(R, r, n_theta=1500, n_direction=1400):
    theta, wt = gauss_nodes(0,2*np.pi,n_theta)
    c=np.cos(theta)
    k1=1/r + 0*c
    k2=c/(R+r*c)
    area_weight=r*(R+r*c)
    P=ellipse_perimeter(k1,k2)
    L=(1/2) * np.sum(wt*area_weight*P)  # azimuth integral 2pi / 4pi
    ang, wa=gauss_nodes(0,2*np.pi,n_direction)
    q = np.abs(k1[:,None]*np.cos(ang)[None,:]**2 +
               k2[:,None]*np.sin(ang)[None,:]**2)
    Lprojected=0.5 * np.sum(wt * area_weight * (q@wa))
    A=4*np.pi**2*R*r
    W=np.pi**2*R**2/(r*np.sqrt(R*R-r*r))
    bound=np.sqrt(0.5*A*W) # torus chi = 0
    return L, Lprojected, A, W, bound


def direct_torus_average(R,r,n_alpha=330,n_phi=850):
    # axial rotational symmetry and u <-> -u symmetry reduce sphere to alpha [0,pi/2]
    alpha, wa=gauss_nodes(0,np.pi/2,n_alpha)
    phi, wp=gauss_nodes(0,2*np.pi,n_phi)
    s=np.sin(alpha)[:,None]
    c=np.cos(alpha)[:,None]
    C=np.cos(phi)[None,:]
    S=np.sin(phi)[None,:]
    theta=np.arctan2(-s*C,c)
    dtheta=s*c*S/(c*c+s*s*C*C)
    total=np.zeros(len(alpha))
    projected=np.zeros(len(alpha))
    for shift in [0,np.pi]:
        t=theta+shift
        ct=np.cos(t); st=np.sin(t)
        azimuth_rad=R+r*ct
        ds=np.sqrt(r*r*dtheta*dtheta+azimuth_rad**2)
        total += ds@wp
        # derivatives in xyz; u=(sin alpha,0,cos alpha)
        # p_theta=(-r sin(t) cos(phi), -r sin(t) sin(phi), r cos(t))
        # p_phi=(-(R+r cos(t))sin(phi), (R+r cos(t))cos(phi),0)
        dx=-r*st*C*dtheta-azimuth_rad*S
        dz=r*ct*dtheta
        udot=s*dx+c*dz
        dp=np.sqrt(np.maximum(ds*ds-udot*udot,0))
        projected += dp@wp
    return np.sum(wa*np.sin(alpha)*total), np.sum(wa*np.sin(alpha)*projected)


def test_sphere(R):
    A=4*np.pi*R*R
    W=4*np.pi
    chi=2
    L=2*np.pi*R
    return L, np.sqrt(0.5*A*(W-np.pi*chi))


# For very high accuracy near a horizontal viewing direction, the contour
# parametrization develops narrow boundary layers near phi=pi/2,3pi/2.
# Use adaptive quadrature for that case instead of a fixed phi grid.
def direct_torus_average_adaptive(R,r,n_alpha=90):
    from scipy.integrate import quad_vec
    alpha,wa=gauss_nodes(0,np.pi/2,n_alpha)
    totals=np.zeros((len(alpha),2))
    for j,aa in enumerate(alpha):
        s=np.sin(aa); c=np.cos(aa)
        def density(phi):
            C=np.cos(phi); S=np.sin(phi)
            theta=np.arctan2(-s*C,c)
            dtheta=s*c*S/(c*c+s*s*C*C)
            vals=np.zeros(2)
            for shift in (0,np.pi):
                t=theta+shift
                ct=np.cos(t); st=np.sin(t)
                radius=R+r*ct
                ds=np.hypot(r*dtheta,radius)
                dx=-r*st*C*dtheta-radius*S
                dz=r*ct*dtheta
                dp=np.sqrt(max(ds*ds-(s*dx+c*dz)**2,0))
                vals += [ds,dp]
            return vals
        totals[j]=quad_vec(density,0,2*np.pi,epsabs=2e-9,epsrel=2e-9,points=[np.pi/2,3*np.pi/2],limit=300)[0]
    return np.sum(wa[:,None]*np.sin(alpha)[:,None]*totals,axis=0)


def main():
    print('Sphere tests (exact formulas)')
    for radius in [0.5, 1, 3]:
        L,bound=test_sphere(radius)
        print(f'R={radius:3.1f}: rim={L:.10f}, bound={bound:.10f}, gap={bound-L:+.3g}')
    print('\nTorus tests (R>r; independent integrations)')
    for R,r in [(2.0,1.0),(np.sqrt(2),1.0),(3.0,0.5)]:
        rim_a,proj_a,A,W,bound=curvature_average_torus(R,r)
        rim_b,proj_b=direct_torus_average_adaptive(R,r,130)
        print(f'R={R:.8f},r={r:.3f}: A={A:.8f}, W={W:.8f}, bound={bound:.8f}')
        print(f'  rim:     curvature={rim_a:.8f}, direct={rim_b:.8f}, difference={rim_b-rim_a:+.4e}')
        print(f'  project: curvature={proj_a:.8f}, direct={proj_b:.8f}, difference={proj_b-proj_a:+.4e}')
        assert rim_a <= bound+1e-8
        assert proj_a <= rim_a+1e-8
        assert abs(rim_a-rim_b)<2.0e-5, 'quadrature resolution mismatch'
        assert abs(proj_a-proj_b)<2.0e-5, 'quadrature resolution mismatch'
    print('\nAll numerical cross-checks passed (finite-resolution consistency checks).')

if __name__=='__main__':
    main()


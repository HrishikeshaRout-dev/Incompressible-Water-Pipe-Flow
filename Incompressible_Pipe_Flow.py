import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

rho   = 998.0
mu    = 1.0e-3
g     = 9.81
U_inf = 0.10
P_inf = 101325.0
D     = 0.05
L     = 2.0
h_fun = lambda x: 0.03 + 0.0*x
nx, ny = 400, 120
T_end, dt = 60.0, 0.02
nu = mu / rho
WALL_TERM = "freestream"

OUT_DIR = "figures"
DPI = 200
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"saved {path}")

x = np.linspace(0.0, L, nx + 1)
h = h_fun(x)
assert np.all(h < D) and D < L
h0 = h[0]
assert np.allclose(h, h0), "this version assumes constant h"
dx = x[1] - x[0]
yv = np.linspace(0.0, h0, ny + 1); dy = yv[1] - yv[0]
Y = np.tile(yv, (nx + 1, 1))

def uyy_wall(u):
    return (2*u[:, 0] - 5*u[:, 1] + 4*u[:, 2] - u[:, 3]) / dy**2

def v_from_continuity(u):
    ux = np.zeros_like(u)
    ux[1:] = (u[1:] - u[:-1]) / dx
    v = np.zeros_like(u)
    v[:, 1:] = -np.cumsum(0.5*(ux[:, 1:] + ux[:, :-1]) * dy, axis=1)
    return ux, v

def rhs(u):
    ux, v = v_from_continuity(u)
    uy  = np.zeros_like(u); uyy = np.zeros_like(u)
    uy[:, 1:-1]  = (u[:, 2:] - u[:, :-2]) / (2*dy)
    uyy[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / dy**2
    Sw = uyy_wall(u)[:, None] if WALL_TERM == "literal" else 0.0
    f = -u*ux - v*uy + nu*(uyy - Sw)
    f[:, 0] = 0.0; f[:, -1] = 0.0; f[0, :] = 0.0
    return f

delta0 = 4*dy
u0 = U_inf*erf(yv/delta0); u0[0] = 0.0
u = np.tile(u0, (nx + 1, 1))

nsteps = int(round(T_end/dt)); hist_t, hist_ut = [], []
for n in range(nsteps):
    k1 = rhs(u); k2 = rhs(u + 0.5*dt*k1); k3 = rhs(u + 0.5*dt*k2); k4 = rhs(u + dt*k3)
    u = u + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    if not np.all(np.isfinite(u)):
        raise RuntimeError(f"blew up at step {n}")
    if n % 25 == 0: hist_t.append(n*dt); hist_ut.append(np.abs(rhs(u)).max())
    if n % 300 == 0 or n == nsteps - 1:
        print(f"t={n*dt:6.2f} s   max|u_t|={np.abs(rhs(u)).max():.3e}   max|u|/U={np.abs(u).max()/U_inf:.3f}")
u_t_final = rhs(u)
ux, v = v_from_continuity(u)
S = uyy_wall(u)

uy  = np.gradient(u, axis=1) / (h[:, None]/ny)
uyy_w = S.copy()
ux  = np.gradient(u, x, axis=0)
uxx = np.gradient(ux, x, axis=0)

tau_w = mu * uy[:, 0]
F_shear = np.concatenate([[0], np.cumsum(0.5*(tau_w[1:] + tau_w[:-1]) * np.diff(x))])

jm = ny // 2
P_x_wall = mu * uyy_w
Pw0_x0 = P_inf + mu*ux[1, jm] + rho*g*Y[0, jm]
P_wall = Pw0_x0 + np.concatenate([[0], np.cumsum(0.5*(P_x_wall[1:] + P_x_wall[:-1]) * np.diff(x))])
P = P_wall[:, None] - rho*g*Y - mu*ux

Px = np.gradient(P, x, axis=0)
Px_eq1 = mu*(np.gradient(uy, axis=1)/(h[:, None]/ny)) - rho*(u*ux + v*uy)
Px_eq3 = P_x_wall[:, None] - mu*uxx

P0w = 0.5*rho*u**2 + P + rho*g*Y

delta = np.array([np.interp(0.99*U_inf, u[i], Y[i]) if u[i].max() >= 0.99*U_inf else np.nan
                  for i in range(nx + 1)])
delta[0] = 0.0

print(f"max delta_w/h = {np.nanmax(delta/h):.3f}   (notes assume delta_w < h)")
print(f"max |v|/U_inf = {np.abs(v).max()/U_inf:.2e}  (notes assume v << U_inf)")
print(f"max |u_yy|wall| = {np.abs(uyy_w[5:]).max():.3e} 1/(m s)")
print(f"total shear force per unit width = {F_shear[-1]:.4e} N/m")

fig, ax = plt.subplots(3, 3, figsize=(17, 13))
ax = ax.ravel()
for xs in [0.05, 0.25, 0.5, 1.0, 1.5, 2.0]:
    i = np.argmin(abs(x - xs))
    ax[0].plot(u[i]/U_inf, Y[i]*1e3, label=f"x={x[i]:.2f} m")
    ax[1].plot(v[i]/U_inf*1e3, Y[i]*1e3, label=f"x={x[i]:.2f} m")
ax[0].set(xlabel="u/U_inf", ylabel="y [mm]", title="Velocity profile u(y)"); ax[0].legend()
ax[1].set(xlabel="v/U_inf x 1e3", ylabel="y [mm]", title="Wall-normal velocity v(y) (continuity)")

cs = ax[2].contourf(x, Y[0]*1e3, (u/U_inf).T, np.linspace(0, 1+1e-9, 31)); plt.colorbar(cs, ax=ax[2])
ax[2].plot(x, delta*1e3, 'w--', label="delta_w"); ax[2].legend()
ax[2].set(xlabel="x [m]", ylabel="y [mm]", title="u/U_inf")

ax[3].plot(x[1:], tau_w[1:]); ax[3].set(xlabel="x [m]", ylabel="tau_w [Pa]", title="Wall shear  mu du/dy|0")
ax[4].plot(x, F_shear); ax[4].set(xlabel="x [m]", ylabel="F [N/m]", title="Cumulative wall shear force")
ax[5].plot(x, delta*1e3, label="delta_w"); ax[5].plot(x, h*1e3, 'k--', label="h(x)")
ax[5].set(xlabel="x [m]", ylabel="mm", title="Boundary-layer thickness"); ax[5].legend()

k0 = 3
ax[6].plot(x[k0:], P_x_wall[k0:]); ax[6].set(xlabel="x [m]", ylabel="dP/dx|0 [Pa/m]", title="Wall pressure gradient = mu u_yy|0")
ax[7].plot(x[k0:], P_wall[k0:] - P_inf); ax[7].set(xlabel="x [m]", ylabel="P|0 - P_inf [Pa]", title="Wall pressure")
cs = ax[8].contourf(x, Y[0]*1e3, (P - P_inf).T, 30); plt.colorbar(cs, ax=ax[8])
ax[8].set(xlabel="x [m]", ylabel="y [mm]", title="P_w - P_inf [Pa]  (P|0 - rho g y - mu u_x)")
for a in ax: a.grid(alpha=.3)
plt.tight_layout()
save(fig, "01_overview.png")

fig2, b = plt.subplots(1, 3, figsize=(16, 4.5))
for xs in [0.5, 1.0, 2.0]:
    i = np.argmin(abs(x - xs))
    b[0].plot(P[i] - P_inf, Y[i]*1e3, label=f"x={x[i]:.2f}")
    b[1].plot(P0w[i] - P_inf, Y[i]*1e3, label=f"x={x[i]:.2f}")
b[0].set(xlabel="P_w - P_inf [Pa]", ylabel="y [mm]", title="Pressure vs y (eq. 2)"); b[0].legend()
b[1].set(xlabel="P_0,w - P_inf [Pa]", ylabel="y [mm]", title="P_0,w = 1/2 rho u^2 + P + rho g y"); b[1].legend()
i = np.argmin(abs(x - 1.0))
b[2].plot(Px_eq1[i], Y[i]*1e3, label="P_x from eq (1)")
b[2].plot(Px_eq3[i], Y[i]*1e3, '--', label="P_x from eq (3)")
b[2].plot(Px[i], Y[i]*1e3, ':', label="d/dx of P(x,y)")
b[2].set(xlabel="dP/dx [Pa/m]", ylabel="y [mm]", title="Consistency of the derived relations, x=1 m"); b[2].legend()
for a in b: a.grid(alpha=.3)
plt.tight_layout()
save(fig2, "02_pressure_consistency.png")

v_bar = v.mean()
h0    = h[0]
yv    = Y[0]
q     = 0.5*rho*U_inf**2
up, vp = u/U_inf, v/v_bar
xp, yp = x/L, yv/h0
Pp    = (P - P_inf)/q
print(f"v_bar (spatial mean of v) = {v_bar:.4e} m/s,  v_bar/U_inf = {v_bar/U_inf:.3e},  h/L = {h0/L:.3e}")
print(f"dynamic pressure 1/2 rho U_inf^2 = {q:.3f} Pa")

fields = [
    (u,       up, "u [m/s]",        "u' = u/U_inf"),
    (v*1e3,   vp, "v [mm/s]",       "v' = v/v_bar"),
    (P-P_inf, Pp, "P - P_inf [Pa]", "P' = (P-P_inf)/(1/2 rho U_inf^2)"),
]

def lev(f, clip=False):
    lo, hi = f.min(), f.max()
    if clip: hi = np.percentile(f[nx//20:], 98)
    return np.linspace(lo, hi + 1e-9*abs(hi) + 1e-12, 41)

fig3, A = plt.subplots(2, 3, figsize=(17, 8))
for k, (fd, fn, ld, ln) in enumerate(fields):
    cl = (k == 1)
    c1 = A[0, k].contourf(x, yv*1e3, fd.T, lev(fd, cl), extend="max" if cl else "neither"); plt.colorbar(c1, ax=A[0, k], label=ld)
    A[0, k].set(xlabel="x [m]", ylabel="y [mm]", title=f"Dimensional: {ld}")
    c2 = A[1, k].contourf(xp, yp, fn.T, lev(fn, cl), extend="max" if cl else "neither"); plt.colorbar(c2, ax=A[1, k], label=ln)
    A[1, k].set(xlabel="x' = x/L", ylabel="y' = y/h", title=f"Nondimensional: {ln}")
plt.tight_layout()
save(fig3, "03_contours_dim_nondim.png")

sx, sy = max(1, nx//100), max(1, ny//50)
XX, YY   = np.meshgrid(x[::sx], yv[::sy], indexing="ij")
XXp, YYp = np.meshgrid(xp[::sx], yp[::sy], indexing="ij")
fig4 = plt.figure(figsize=(18, 10))
for k, (fd, fn, ld, ln) in enumerate(fields):
    a1 = fig4.add_subplot(2, 3, k+1, projection="3d")
    a1.plot_surface(XX[3:], YY[3:]*1e3, fd[::sx, ::sy][3:], cmap="viridis", linewidth=0)
    a1.set(xlabel="x [m]", ylabel="y [mm]", zlabel=ld, title=f"Dimensional: {ld}")
    a2 = fig4.add_subplot(2, 3, k+4, projection="3d")
    a2.plot_surface(XXp[3:], YYp[3:], fn[::sx, ::sy][3:], cmap="viridis", linewidth=0)
    a2.set(xlabel="x'", ylabel="y'", zlabel=ln.split(" = ")[0], title=f"Nondimensional: {ln}")
plt.tight_layout()
save(fig4, "04_surfaces_3d.png")

fig5, B = plt.subplots(1, 3, figsize=(17, 4.8))
for xs in [0.025, 0.125, 0.25, 0.5, 0.75, 1.0]:
    i = np.argmin(abs(xp - xs))
    B[0].plot(up[i], yp, label=f"x'={xp[i]:.3f}")
    B[1].plot(vp[i], yp, label=f"x'={xp[i]:.3f}")
    B[2].plot(Pp[i], yp, label=f"x'={xp[i]:.3f}")
B[0].set(xlabel="u'", ylabel="y'", title="u'(y')"); B[0].legend()
B[1].set(xlabel="v'", ylabel="y'", title="v'(y')")
B[2].set(xlabel="P'", ylabel="y'", title="P'(y')")
for a in B: a.grid(alpha=.3)
plt.tight_layout()
save(fig5, "05_nondim_profiles.png")

fig6, C = plt.subplots(figsize=(6, 4))
C.semilogy(hist_t, hist_ut); C.set(xlabel="t [s]", ylabel="max |u_t| [m/s^2]", title="Unsteady term u_t of eq. [1]"); C.grid(alpha=.3)
plt.tight_layout()
save(fig6, "06_unsteady_history.png")

plt.show()
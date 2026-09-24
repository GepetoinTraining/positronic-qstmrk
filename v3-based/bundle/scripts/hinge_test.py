"""
Hinge test — can the onset phase of a noise-born oscillator be read
back from its shadow (the real-part graph), and how far later?

Model: Stuart–Landau, the normal form of any oscillator switching on
(a Hopf onset). z is the live object; x = Re z is the graph we'd record.
    dz = [(mu + i*w) z - |z|^2 z] dt + sigma dW      (complex noise)
Noise source: os.urandom (kernel entropy of this container), Box–Muller.
Instrument noise added to x before any analysis.
"""
import os, numpy as np
from scipy.signal import hilbert
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def urandom_normal(shape):
    n = int(np.prod(shape))
    u = np.frombuffer(os.urandom(8 * 2 * n), dtype=np.uint64).astype(np.float64)
    u = (u + 0.5) / 2.0**64
    u1, u2 = u[:n], u[n:]
    return (np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)).reshape(shape)

TRIALS, SPC, CYCLES = 400, 100, 70          # samples per cycle, cycles
dt, w = 1.0 / SPC, 2 * np.pi
ON = 10 * SPC                                # 10 cycles below threshold, then switch on
N = SPC * CYCLES
t = np.arange(N) * dt
Q = SPC // 4                                 # quarter turn = the i, as a sample offset

def run(sigma, meas):
    z = np.zeros(TRIALS, complex)
    Z = np.empty((N, TRIALS), complex)
    dW = (urandom_normal((N, TRIALS)) + 1j * urandom_normal((N, TRIALS))) * np.sqrt(dt / 2)
    for k in range(N):
        mu = -0.5 if k < ON else 0.5            # current crosses threshold at ON
        z = z + ((mu + 1j * w) * z - (abs(z) ** 2) * z) * dt + sigma * dW[k]
        Z[k] = z
    X = Z.real + meas * urandom_normal((N, TRIALS))      # the shadow, as measured
    H = hilbert(X, axis=0)                                # global un-collapse (FFT)
    # quarter-turn lookup: Im z(t) = x(t - T/4) for a clean rotation. No arithmetic,
    # just reading the sample one quarter-cycle back: the {1,i,-1,-i} step as memory.
    D = np.zeros_like(Z); D[Q:] = X[Q:] + 1j * X[:-Q]
    return Z, X, H, D

def demod(v, k):                     # strip the known rotation w*t
    return np.angle(v * np.exp(-1j * w * t[k]))

def analyse(sigma, meas):
    Z, X, H, D = run(sigma, meas)
    HINGE = 10 * sigma                                   # hinge = clear of the noise floor
    idx = np.arange(TRIALS)
    kt = np.argmax(np.abs(Z) > HINGE, axis=0)            # true hinge
    phi_true = demod(Z[kt, idx], kt)
    out = dict(Z=Z, X=X, H=H, D=D, kt=kt, hinge=HINGE)
    for name, A in (("hilbert", H), ("quarter", D)):
        ks = np.argmax(np.abs(A) > HINGE, axis=0)
        out[name + "_dk"] = ks - kt
        out[name + "_err"] = np.angle(np.exp(1j * (demod(A[ks, idx], ks) - phi_true)))
    lags = np.arange(0, 41)                              # cycles after hinge
    R = []                                               # does the hinge phase survive?
    for L in lags:
        k = np.minimum(kt + L * SPC, N - 1)
        R.append(abs(np.mean(np.exp(1j * (demod(D[k, idx], k) - phi_true)))))
    out.update(lags=lags, R=np.array(R))
    return out

res = {s: analyse(s, 1e-4) for s in (1e-3, 3e-2)}
deg = lambda e: np.degrees(np.median(np.abs(e)))
for s, r in res.items():
    print(f"sigma={s:g} hinge={r['hinge']:g}")
    for m in ("hilbert", "quarter"):
        print(f"  {m:8s} hinge timing median |dk| {np.median(np.abs(r[m+'_dk'])):.0f} samples, "
              f"phase err median {deg(r[m+'_err']):.1f} deg")
    print(f"  rewind R at 0/5/10/40 cycles: " + "/".join(f"{r['R'][L]:.3f}" for L in (0,5,10,40)))

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(3, 1, figsize=(8, 11))
r = res[1e-3]; j = 0; k0 = r["kt"][j]
win = slice(max(k0 - 7 * SPC, 0), k0 + 4 * SPC)
ax[0].plot(t[win], r["X"][win, j], color="#4C72B0", lw=1, label="shadow x = Re z (the graph)")
ax[0].plot(t[win], np.abs(r["D"][win, j]), color="#DD8452", ls="--", lw=2, label="|x + i·x(t−T/4)| quarter-turn rebuild")
ax[0].plot(t[win], np.abs(r["H"][win, j]), color="0.4", ls=":", lw=1.5, label="|Hilbert| global rebuild")
ax[0].axvline(t[k0], color="k", ls="-.", lw=1, label="true hinge (10σ)")
ax[0].axvline(t[ON], color="0.5", ls="-", lw=1, label="drive crosses threshold")
ax[0].set_yscale("symlog", linthresh=1e-3)
ax[0].set_title("One onset: noise, then the first turn that grows")
ax[0].set_xlabel("time (cycles)"); ax[0].legend(fontsize=8, loc="lower right")
bins = np.linspace(-180, 180, 37)
ax[1].hist(np.degrees(r["hilbert_err"]), bins=bins, color="0.6", alpha=.7, hatch="..", label="global Hilbert")
ax[1].hist(np.degrees(r["quarter_err"]), bins=bins, color="#DD8452", alpha=.7, label="quarter-turn lookup")
ax[1].set_title("Hinge phase read from the shadow: error vs true (σ=1e-3)")
ax[1].set_xlabel("degrees"); ax[1].legend()
for (s, rr), c, ls in zip(res.items(), ("#4C72B0", "#DD8452"), ("-", "--")):
    ax[2].plot(rr["lags"], rr["R"], color=c, ls=ls, lw=2, label=f"noise σ={s:g}")
ax[2].set_ylim(0, 1.05)
ax[2].set_title("Rewind: hinge phase still readable N cycles later")
ax[2].set_xlabel("cycles after hinge"); ax[2].set_ylabel("coherence R (1 = exact)")
ax[2].legend()
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/hinge_test.png", dpi=130)

# ---- backward: rewind from the final graph into the flat region ----
print("\nbackward from final phase (hinge+40 cycles) into the pre-hinge noise:")
for s, r in res.items():
    Z, D, kt = r["Z"], r["D"], r["kt"]; idx = np.arange(TRIALS)
    kf = np.minimum(kt + 40 * SPC, N - 1); phi_f = demod(D[kf, idx], kf)
    row = []
    for L in (-8, -6, -4, -3, -2, -1, 0):
        k = np.clip(kt + L * SPC, Q, N - 1)
        amp = np.median(np.abs(Z[k, idx]))
        Rb = abs(np.mean(np.exp(1j * (demod(D[k, idx], k) - phi_f))))
        row.append(f"{L:+d}c amp {amp:.1e} R {Rb:.2f}")
    print(f" sigma={s:g}: " + " | ".join(row))

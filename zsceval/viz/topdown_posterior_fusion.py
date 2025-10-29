import math
import numpy as np

EPS = 1e-9

def normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, EPS, None)
    s = P.sum()
    if s <= 0:
        return np.ones_like(P) / P.size
    return P / s

def entropy(P: np.ndarray) -> float:
    Pn = normalize(P).ravel()
    return float(-(Pn * np.log(Pn)).sum())

def gaussian_kernel2d(sigma: float, radius: int = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([[1.0]], dtype=float)
    if radius is None:
        radius = max(1, int(3.0 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=float)
    ys = np.arange(-radius, radius + 1, dtype=float)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    K = np.exp(- (X**2 + Y**2) / (2 * sigma**2))
    K /= K.sum()
    return K

def convolve_same(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    # zero-padding convolution, "same" output size
    R, C = P.shape
    kr, kc = K.shape
    pr, pc = kr // 2, kc // 2
    out = np.zeros_like(P, dtype=float)
    # pad
    Pp = np.pad(P, ((pr, pr), (pc, pc)), mode="edge")
    for i in range(R):
        for j in range(C):
            region = Pp[i:i+kr, j:j+kc]
            out[i, j] = float((region * K).sum())
    return out

def shift_with_wrap(P: np.ndarray, dr: int, dc: int) -> np.ndarray:
    # wrap-around shift; for clamped use edge padding + slice, but wrap is OK for grids
    return np.roll(np.roll(P, dr, axis=0), dc, axis=1)

def build_ior_mask(shape, fixations, sigma=1.0, strength=0.10, k_last=3):
    if not fixations:
        return np.ones(shape, dtype=float)
    R, C = shape
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    field = np.zeros(shape, dtype=float)
    for (r, c) in fixations[-k_last:]:
        field += np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma ** 2))
    if field.max() > 0:
        field = field / field.max()
    # multiplicative soft mask
    mask = 1.0 - strength * field
    return np.clip(mask, 0.0, 1.0)

def predict_prior(A_prev: np.ndarray,
                  sigma: float = 1.0,
                  drift: tuple = (0, 0),
                  ior_mask: np.ndarray = None) -> np.ndarray:
    """One-step predicted prior π_t from previous posterior A_{t-1}."""
    K = gaussian_kernel2d(sigma=sigma)
    P = convolve_same(A_prev, K)
    if drift != (0, 0):
        P = shift_with_wrap(P, int(drift[0]), int(drift[1]))
    if ior_mask is not None:
        P = P * ior_mask
    return normalize(P)

def fuse_log(A_post_hat: np.ndarray, prior_pi: np.ndarray, eta: float = 0.4) -> np.ndarray:
    """Log-weighted fusion to avoid double counting and keep stability."""
    A_post_hat = normalize(A_post_hat)
    prior_pi   = normalize(prior_pi)
    logA = (1 - eta) * np.log(A_post_hat + EPS) + eta * np.log(prior_pi + EPS)
    A = np.exp(logA - logA.max())
    return normalize(A)

def fuse_dirichlet(A_post_hat: np.ndarray, prior_pi: np.ndarray, kappa: float = 4.0, N: float = 1.0) -> np.ndarray:
    """Pseudo-count (Dirichlet-like) averaging."""
    A_post_hat = normalize(A_post_hat)
    prior_pi   = normalize(prior_pi)
    return normalize(N * A_post_hat + kappa * prior_pi)

class AttentionFuser:
    """Maintains state and performs posterior-prior fusion per frame for an 8x5 (or any) grid,
       with short-term visual memory (STM) boosting."""
    def __init__(self,
                 shape=(8,5),
                 fusion="log",           # "log" or "dirichlet"
                 eta=0.4,                # log fusion weight (used if adaptive off)
                 kappa=4.0, N=1.0,       # dirichlet params
                 sigma=1.0,              # gaussian blur for prior prediction
                 ior_sigma=1.0,          # IOR spread
                 ior_strength=0.10,      # IOR multiplicative subtraction
                 ior_k=3,                # last K fixations for IOR
                 momentum=True,
                 momentum_scale=1,       # drift of 1 grid cell by default
                 eta_min=0.05, eta_max=0.9,
                 # --- Short-term memory params ---
                 stm_capacity=4,
                 stm_tau=5.0,            # decay time constant (in frames)
                 stm_sigma=1.2,          # spatial spread of STM boost
                 eta_stm=0.2             # weight of STM boost (0.1~0.3 recommended)
                 ):

        self.shape = shape
        self.fusion = fusion
        self.eta = eta
        self.kappa = kappa
        self.N = N
        self.sigma = sigma
        self.ior_sigma = ior_sigma
        self.ior_strength = ior_strength
        self.ior_k = ior_k
        self.momentum = momentum
        self.momentum_scale = momentum_scale
        self.eta_min, self.eta_max = eta_min, eta_max

        # states
        self.A_prev = np.ones(shape, dtype=float) / (shape[0] * shape[1])
        self.fixations = []  # list of (r,c)
        self.hitrate_ma = 0.0  # moving average of hit (if provided)
        self.last_fix = None
        self.prev_fix = None

        # STM (short-term memory)
        self.stm = []  # list of dicts: {"p":(r,c), "s":strength, "t":frame_idx}
        self.stm_capacity = stm_capacity
        self.stm_tau = stm_tau
        self.stm_sigma = stm_sigma
        self.eta_stm = eta_stm
        self._frame = 0

    # --------- STM helpers ---------
    def _stm_boost(self) -> np.ndarray:
        """Return multiplicative boost map B_t (>=1), from STM slots with exponential decay."""
        R, C = self.shape
        if len(self.stm) == 0:
            return np.ones((R, C), dtype=float)
        rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
        B = np.zeros((R, C), dtype=float)
        for it in self.stm:
            age = max(0, self._frame - it["t"])
            s_eff = it["s"] * np.exp(-age / max(1e-6, self.stm_tau))
            if s_eff <= 1e-6:
                continue
            pr, pc = it["p"]
            spatial = np.exp(-((rr - pr)**2 + (cc - pc)**2) / (2 * self.stm_sigma**2))
            B += s_eff * spatial
        if B.max() > 0:
            B = 1.0 + (B / (B.max() + 1e-9))  # soft boost centered at 1
        else:
            B = np.ones((R, C), dtype=float)
        return B

    def _stm_encode(self, fixation_rc, boost=1.0):
        """Insert/update STM slot near current fixation."""
        r, c = fixation_rc
        # update near slot if exists
        for it in self.stm:
            pr, pc = it["p"]
            if abs(pr - r) + abs(pc - c) <= 1:
                it["s"] = min(1.0, it["s"] + 0.3 * boost)
                it["t"] = self._frame
                break
        else:
            # insert new
            self.stm.append({"p": (r, c), "s": 0.6 * boost, "t": self._frame})
            if len(self.stm) > self.stm_capacity:
                # drop weakest
                self.stm.sort(key=lambda z: z["s"])
                self.stm.pop(0)
        # prune decayed
        cleaned = []
        for it in self.stm:
            age = max(0, self._frame - it["t"])
            s_eff = it["s"] * np.exp(-age / max(1e-6, self.stm_tau))
            if s_eff > 1e-3:
                cleaned.append(it)
        self.stm = cleaned

    # --------- Core helpers ---------
    def _drift_from_momentum(self):
        if not self.momentum or len(self.fixations) < 2:
            return (0, 0)
        r1, c1 = self.fixations[-1]
        r0, c0 = self.fixations[-2]
        dr = int(np.sign(r1 - r0)) * self.momentum_scale
        dc = int(np.sign(c1 - c0)) * self.momentum_scale
        return (dr, dc)

    def adapt_eta(self, A_post_hat: np.ndarray):
        H = entropy(A_post_hat)  # higher H => more weight to prior
        # normalize H roughly by log(#cells)
        Hmax = math.log(A_post_hat.size + EPS)
        x = 1.0 - (H / (Hmax + EPS))   # confidence proxy in [0,1]
        # more uncertain (x small) => eta larger
        eta = self.eta_min + (self.eta_max - self.eta_min) * (1.0 - x)
        return float(np.clip(eta, self.eta_min, self.eta_max))

    # --------- Main step ---------
    def step(self, A_post_hat: np.ndarray, hit: float = None, use_adaptive_eta=False):
        """Fuse current posterior (already observed) with predicted prior and STM; update fixation & state.

        Args:
            A_post_hat: np.ndarray, shape=self.shape, the observed posterior for this frame.
            hit: optional float in [0,1] indicating success at this frame (used only for logging/eta heuristics).
            use_adaptive_eta: whether to adapt eta by entropy.

        Returns:
            A_t: fused posterior (final)
            fix_rc: sampled fixation (row,col) from A_t
            prior_pi: predicted prior (without STM) used for fusion
            eta_used: eta weight applied to prior in log-fusion
        """
        assert A_post_hat.shape == self.shape
        # IOR
        ior_mask = build_ior_mask(self.shape, self.fixations,
                                  sigma=self.ior_sigma,
                                  strength=self.ior_strength,
                                  k_last=self.ior_k)
        # Momentum drift
        drift = self._drift_from_momentum()

        # Predict prior (from previous A)
        prior_pi = predict_prior(self.A_prev, sigma=self.sigma, drift=drift, ior_mask=ior_mask)

        # Short-term memory boost
        B_t = self._stm_boost()  # multiplicative map >= 1

        A_post_hat = normalize(np.clip(A_post_hat, 0, None) ** 0.85)
        
        # Fuse
        eta_used = self.adapt_eta(A_post_hat) if use_adaptive_eta else self.eta
        if self.fusion == "log":
            # log-fusion + STM term
            A_post_hat = normalize(A_post_hat)
            prior_pi   = normalize(prior_pi)
            logA = (1 - eta_used) * np.log(A_post_hat + EPS) + eta_used * np.log(prior_pi + EPS) \
                   + self.eta_stm * np.log(B_t + EPS)
            A_t = np.exp(logA - logA.max())
            A_t = normalize(A_t)
        else:
            # dirichlet fusion then apply STM multiplicatively
            A_t = fuse_dirichlet(A_post_hat, prior_pi, kappa=self.kappa, N=self.N)
            A_t = normalize(A_t * (B_t ** self.eta_stm))

        # Sample fixation (soft) — use argmax if you want pure exploitation
        flat = A_t.ravel()
        idx = np.random.choice(flat.size, p=flat)
        r, c = np.unravel_index(idx, self.shape)

        # Update traces
        self.fixations.append((r, c))
        if len(self.fixations) > 32:
            self.fixations.pop(0)
        self.prev_fix = self.last_fix
        self.last_fix = (r, c)
        if hit is not None:
            self.hitrate_ma = 0.9 * self.hitrate_ma + 0.1 * hit

        # --- STM encode after choosing fixation ---
        self._stm_encode((r, c), boost=1.0)

        # Roll state and frame
        self.A_prev = A_t.copy()
        self._frame += 1

        return A_t, (r, c), prior_pi, eta_used

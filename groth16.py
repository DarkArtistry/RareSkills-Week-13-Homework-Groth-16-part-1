"""
Groth16 (FULL) — RareSkills Homework 12 (Week 13)
===================================================
Implements the COMPLETE Groth16 proving system end-to-end:
  R1CS -> QAP -> Trusted Setup (multi-party) -> Prover -> Verifier

This upgrades the Part 1 implementation to include:
  * gamma (gamma) — separates verifier's public-input term
  * delta (delta) — separates prover's private-input + H(t)/T term
  * r, s          — prover's per-proof randomness (zero-knowledge blinding)
  * Public/private witness split (first `num_public` columns are public)

Reference: https://www.rareskills.io/post/groth16 (formulas at the end)

Communication flow:
  1. QAP is derived from the public circuit (R1CS).
  2. Trusted Setup Ceremony:
       Phase 1 (multi-party Powers of Tau):
         Alice ---tau_1---> [SRS] ---tau_2---> Bob ---tau_3---> [SRS]
         (deletes tau_1)              (deletes tau_2)              (deletes tau_3)
       Phase 2 (coordinator):
         Generates alpha, beta, gamma, delta, computes circuit-specific
         SRS elements (eta, Psi_public, Psi_private), then DESTROYS all four.
     Result: public SRS (only EC points — all scalars destroyed).
  3. Prover receives: SRS + witness (private).
     Prover picks fresh random r, s (destroyed after use).
     Prover computes proof = (A1, B2, C1) and sends it to Verifier.
  4. Verifier receives: SRS + proof + the PUBLIC part of the witness.
     Verifier does NOT see the private witness.
     Verifier checks:
       e(-[A]1, [B]2) * e([alpha]1, [beta]2) * e([X]1, [gamma]2) * e([C]1, [delta]2) = I_12
     where [X]1 = sum_{i in public} a_i * Psi_public_i.

Features:
  - Dynamic R1CS: user-provided or default (out = 3x^2*y + 5xy - x - 2y + 3)
  - Two Lagrange interpolation modes: manual (from scratch) and library (galois)
  - Multi-party trusted setup (Phase 1) demonstrating Powers of Tau security
  - Public/private input separation via gamma / delta

Usage:
  python groth16.py
"""

import numpy as np
import secrets
import galois
from functools import reduce

from py_ecc.bn128 import (
    G1, G2, multiply, add, pairing, curve_order, neg, eq, Z1, Z2,
    final_exponentiate,
)
from py_ecc.bn128.bn128_curve import FQ12


# ============================================================
#  CONSTANTS
# ============================================================

p = curve_order
GF = galois.GF(p, primitive_element=5, verify=False)


# ============================================================
#  FINITE FIELD ARITHMETIC (mod p)
# ============================================================

def f_add(a, b): return (a + b) % p
def f_sub(a, b): return (a - b) % p
def f_mul(a, b): return (a * b) % p
def f_neg(a):    return (-a) % p

def f_inv(a):
    """Multiplicative inverse via Fermat's little theorem: a^{-1} = a^{p-2} mod p"""
    assert a % p != 0, "Cannot invert zero"
    return pow(a, p - 2, p)

def f_div(a, b): return f_mul(a, f_inv(b))


# ============================================================
#  POLYNOMIAL ARITHMETIC
#  Ascending coefficient order: [a0, a1, a2, ...] = a0 + a1*x + a2*x^2 + ...
# ============================================================

def poly_strip(coeffs):
    """Remove trailing zero coefficients (high-degree zeros)."""
    result = list(coeffs)
    while len(result) > 1 and result[-1] % p == 0:
        result.pop()
    return result

def poly_add(a, b):
    size = max(len(a), len(b))
    result = [0] * size
    for i in range(len(a)): result[i] = f_add(result[i], a[i])
    for i in range(len(b)): result[i] = f_add(result[i], b[i])
    return poly_strip(result)

def poly_sub(a, b):
    size = max(len(a), len(b))
    result = [0] * size
    for i in range(len(a)): result[i] = f_add(result[i], a[i])
    for i in range(len(b)): result[i] = f_sub(result[i], b[i])
    return poly_strip(result)

def poly_mul(a, b):
    """Schoolbook multiplication: (a_i * x^i)(b_j * x^j) -> coeff at index i+j."""
    if not a or not b: return [0]
    result = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] = f_add(result[i + j], f_mul(a[i], b[j]))
    return poly_strip(result)

def poly_scalar_mul(poly, scalar):
    return poly_strip([f_mul(c, scalar) for c in poly])

def poly_eval(poly, x):
    """Evaluate polynomial at x using Horner's method."""
    x = x % p
    result = 0
    for i in range(len(poly) - 1, -1, -1):
        result = f_add(f_mul(result, x), poly[i])
    return result

def poly_div(num, den):
    """Polynomial long division -> (quotient, remainder) in ascending order."""
    num = poly_strip([c % p for c in num])
    den = poly_strip([c % p for c in den])
    if len(num) < len(den):
        return [0], num
    num_d = list(reversed(num))
    den_d = list(reversed(den))
    q_len = len(num) - len(den) + 1
    quot = [0] * q_len
    for i in range(q_len):
        quot[i] = f_div(num_d[i], den_d[0])
        for j in range(len(den_d)):
            num_d[i + j] = f_sub(num_d[i + j], f_mul(quot[i], den_d[j]))
    quot.reverse()
    rem = list(reversed(num_d[q_len:]))
    return poly_strip(quot), poly_strip(rem if rem else [0])

def poly_degree(poly):
    return len(poly_strip(poly)) - 1

def poly_to_str(poly):
    """Pretty-print a polynomial (ascending coefficients)."""
    stripped = poly_strip(poly)
    if all(c % p == 0 for c in stripped):
        return "0"
    terms = []
    for i in range(len(stripped) - 1, -1, -1):
        c = stripped[i] % p
        if c == 0:
            continue
        if c > p // 2:
            c_display, sign = p - c, ("-" if not terms else " - ")
        else:
            c_display, sign = c, ("" if not terms else " + ")
        if i == 0:
            terms.append(f"{sign}{c_display}")
        elif i == 1:
            terms.append(f"{sign}{'' if c_display == 1 else c_display}x")
        else:
            terms.append(f"{sign}{'' if c_display == 1 else c_display}x^{i}")
    return "".join(terms) if terms else "0"


# ============================================================
#  LAGRANGE INTERPOLATION (two modes)
# ============================================================

def lagrange_manual(xs, ys):
    """
    Manual Lagrange interpolation over F_p.
    For each point j, build basis polynomial L_j(x) = prod_{i!=j} (x - xs[i]) / (xs[j] - xs[i]),
    then P(x) = sum_j ys[j] * L_j(x).
    Returns ascending coefficients.
    """
    n = len(xs)
    if all(y % p == 0 for y in ys):
        return [0]
    result = [0]
    for j in range(n):
        if ys[j] % p == 0:
            continue
        num_poly = [1]
        den_scalar = 1
        for i in range(n):
            if i == j:
                continue
            num_poly = poly_mul(num_poly, [f_neg(xs[i]), 1])
            den_scalar = f_mul(den_scalar, f_sub(xs[j], xs[i]))
        scale = f_mul(ys[j], f_inv(den_scalar))
        result = poly_add(result, poly_scalar_mul(num_poly, scale))
    return poly_strip(result)


def lagrange_library(xs, ys):
    """
    Lagrange interpolation using galois library.
    Returns ascending coefficients for comparison with manual mode.
    """
    xs_gf = GF([x % p for x in xs])
    ys_gf = GF([y % p for y in ys])
    poly = galois.lagrange_poly(xs_gf, ys_gf)
    return poly_strip(list(reversed([int(c) for c in poly.coeffs])))


# ============================================================
#  R1CS INPUT (dynamic: default or custom)
# ============================================================

def get_default_r1cs():
    """
    Default R1CS for: out = 3x^2*y + 5xy - x - 2y + 3
    Flattened into 3 multiplication gates:
      Gate 1: (3x) * (x) = v1
      Gate 2: (v1) * (y) = v2
      Gate 3: (x) * (5y) = -3 + out + x + 2y - v2

    Witness: [1, out, x, y, v1, v2]  with x=100, y=100
    """
    A = np.array([[0, 0, 3, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0, 0]])

    B = np.array([[0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 5, 0, 0]])

    C = np.array([[ 0, 0, 0, 0, 1, 0],
                   [ 0, 0, 0, 0, 0, 1],
                   [-3, 1, 1, 2, 0,-1]])

    x_val, y_val = 100, 100
    v1 = 3 * x_val * x_val
    v2 = v1 * y_val
    out = 3 * x_val**2 * y_val + 5 * x_val * y_val - x_val - 2 * y_val + 3
    witness = np.array([1, out, x_val, y_val, v1, v2])
    return A, B, C, witness


def get_custom_r1cs():
    """Prompt user for custom R1CS matrices and witness."""
    print("\n  Enter custom R1CS:")
    n = int(input("    Number of constraints (rows): "))
    m = int(input("    Number of variables (columns): "))

    print(f"\n  Matrix A ({n}x{m}), row by row (space-separated integers):")
    A = np.array([list(map(int, input(f"    Row {i+1}: ").split())) for i in range(n)])

    print(f"\n  Matrix B ({n}x{m}):")
    B = np.array([list(map(int, input(f"    Row {i+1}: ").split())) for i in range(n)])

    print(f"\n  Matrix C ({n}x{m}):")
    C = np.array([list(map(int, input(f"    Row {i+1}: ").split())) for i in range(n)])

    print(f"\n  Witness ({m} values, first element should be 1):")
    witness = np.array(list(map(int, input("    Witness: ").split())))

    return A, B, C, witness


def validate_r1cs(A, B, C, witness):
    """Verify all R1CS constraints: (A_i . w) * (B_i . w) = (C_i . w) for each gate i."""
    Aw = A.dot(witness)
    Bw = B.dot(witness)
    Cw = C.dot(witness)
    all_pass = True
    for i in range(A.shape[0]):
        ok = (Aw[i] * Bw[i] == Cw[i])
        status = "PASS" if ok else "FAIL"
        print(f"    Gate {i+1}: ({Aw[i]}) * ({Bw[i]}) = {Aw[i]*Bw[i]} ?= {Cw[i]}  [{status}]")
        if not ok:
            all_pass = False
    return all_pass


# ============================================================
#  QAP (Quadratic Arithmetic Program)
# ============================================================

class QAP:
    """
    Converts R1CS (A, B, C, witness) into QAP polynomials.

    For each column k, interpolate through the gate values to get u_k(x), v_k(x), w_k(x).
    Then combine with witness: U(x) = sum a_k * u_k(x), etc.
    Compute H(x) = (U*V - W) / T where T(x) = (x-1)(x-2)...(x-n).
    """

    def __init__(self, A, B, C, witness, lagrange_mode="manual"):
        self.n = A.shape[0]      # number of constraints (rows)
        self.m = A.shape[1]      # number of columns (variables)
        self.witness = witness
        self.lagrange_mode = lagrange_mode
        interp = lagrange_manual if lagrange_mode == "manual" else lagrange_library

        xs = list(range(1, self.n + 1))

        # Interpolate each column into polynomials
        self.U_polys = []  # u_k(x) from A columns
        self.V_polys = []  # v_k(x) from B columns
        self.W_polys = []  # w_k(x) from C columns
        for k in range(self.m):
            ys_a = [int(A[i][k]) % p for i in range(self.n)]
            ys_b = [int(B[i][k]) % p for i in range(self.n)]
            ys_c = [int(C[i][k]) % p for i in range(self.n)]
            self.U_polys.append(interp(xs, ys_a))
            self.V_polys.append(interp(xs, ys_b))
            self.W_polys.append(interp(xs, ys_c))

        # Target polynomial T(x) = (x-1)(x-2)...(x-n)
        self.T = [1]
        for r in xs:
            self.T = poly_mul(self.T, [f_neg(r), 1])

        # Combine with witness: U(x) = sum a_k * u_k(x), etc.
        self.U = [0]
        self.V = [0]
        self.W = [0]
        for k in range(self.m):
            wk = int(witness[k]) % p
            if wk == 0:
                continue
            self.U = poly_add(self.U, poly_scalar_mul(self.U_polys[k], wk))
            self.V = poly_add(self.V, poly_scalar_mul(self.V_polys[k], wk))
            self.W = poly_add(self.W, poly_scalar_mul(self.W_polys[k], wk))

        # H(x) = (U*V - W) / T — must divide evenly
        uv_minus_w = poly_sub(poly_mul(self.U, self.V), self.W)
        self.H, remainder = poly_div(uv_minus_w, self.T)
        assert all(c % p == 0 for c in remainder), \
            "QAP not satisfied: (U*V - W) is not divisible by T!"

    def verify(self):
        """Verify QAP identity: U(x)*V(x) = W(x) + H(x)*T(x) as polynomials."""
        lhs = poly_mul(self.U, self.V)
        rhs = poly_add(self.W, poly_mul(self.H, self.T))
        lhs_s, rhs_s = poly_strip(lhs), poly_strip(rhs)
        max_len = max(len(lhs_s), len(rhs_s))
        lp = lhs_s + [0] * (max_len - len(lhs_s))
        rp = rhs_s + [0] * (max_len - len(rhs_s))
        return all((a - b) % p == 0 for a, b in zip(lp, rp))

    def print_polynomials(self):
        """Display all QAP column polynomials."""
        print(f"\n  U polynomials (from A — left inputs):")
        for k in range(self.m):
            print(f"    u_{k}(x) = {poly_to_str(self.U_polys[k])}")
        print(f"\n  V polynomials (from B — right inputs):")
        for k in range(self.m):
            print(f"    v_{k}(x) = {poly_to_str(self.V_polys[k])}")
        print(f"\n  W polynomials (from C — outputs):")
        for k in range(self.m):
            print(f"    w_{k}(x) = {poly_to_str(self.W_polys[k])}")
        print(f"\n  Combined with witness:")
        print(f"    U(x) = {poly_to_str(self.U)}")
        print(f"    V(x) = {poly_to_str(self.V)}")
        print(f"    W(x) = {poly_to_str(self.W)}")
        print(f"    H(x) = {poly_to_str(self.H)}")
        print(f"    T(x) = {poly_to_str(self.T)}")


# ============================================================
#  ELLIPTIC CURVE HELPERS
# ============================================================

def inner_product_ec(scalars, points, identity=Z1):
    """Compute sum( scalar_i * point_i ) on the elliptic curve."""
    result = None
    for s, pt in zip(scalars, points):
        s = int(s) % curve_order
        if s == 0:
            continue
        term = multiply(pt, s)
        result = term if result is None else add(result, term)
    return result if result is not None else identity


def point_str(pt):
    """Short string representation of an EC point."""
    if pt is None or pt == Z1 or pt == Z2:
        return "O (infinity)"
    if isinstance(pt[0], tuple):
        return f"G2({str(pt[0][0])[:12]}...)"
    return f"G1({str(pt[0])[:12]}..., {str(pt[1])[:12]}...)"


# ============================================================
#  MULTI-PARTY TRUSTED SETUP
#
#  The Groth16 trusted setup uses a "Powers of Tau" ceremony
#  where multiple independent participants each contribute
#  their own secret randomness. The combined secret is
#  tau = tau_1 * tau_2 * ... * tau_k (mod p).
#
#  Security guarantee: as long as AT LEAST ONE participant
#  honestly deletes their secret, the combined tau is unknown
#  — even if all other participants collude.
#
#  This is because knowing tau requires knowing ALL factors,
#  and the discrete log problem prevents extracting individual
#  secrets from the published EC points.
# ============================================================

class CeremonyParticipant:
    """
    One participant in the Powers of Tau ceremony.

    Each participant:
      1. Generates a private secret tau_i
      2. Updates the SRS by multiplying each power: srs[j] *= tau_i^j
      3. DESTROYS their secret — the critical security step

    The participant NEVER learns the previous combined tau.
    They only see EC points (discrete log is intractable)
    and blindly multiply by their own tau_i powers.
    """

    def __init__(self, name):
        self.name = name
        self._tau = secrets.randbelow(p - 1) + 1
        self._destroyed = False

    def contribute(self, powers_g1, powers_g2):
        """
        Update the SRS with this participant's secret.

        The previous SRS encodes some combined tau_prev:
          powers_g1 = [tau_prev^0 * G1, tau_prev^1 * G1, tau_prev^2 * G1, ...]

        We multiply each element by our tau_i raised to the matching power:
          new[j] = tau_i^j * old[j] = (tau_i * tau_prev)^j * G1

        After this, the effective tau becomes tau_i * tau_prev.
        We never learn tau_prev (it's hidden inside EC points).
        """
        assert not self._destroyed, f"{self.name}: secret already destroyed!"

        # Update G1 powers: srs1[j] = tau_i^j * srs1[j]
        new_g1 = []
        tau_pow = 1
        for pt in powers_g1:
            new_g1.append(multiply(pt, tau_pow))
            tau_pow = f_mul(tau_pow, self._tau)

        # Update G2 powers: srs2[j] = tau_i^j * srs2[j]
        new_g2 = []
        tau_pow = 1
        for pt in powers_g2:
            new_g2.append(multiply(pt, tau_pow))
            tau_pow = f_mul(tau_pow, self._tau)

        return new_g1, new_g2

    def destroy_secret(self):
        """
        DESTROY this participant's secret forever.

        Once destroyed, our tau_i is unrecoverable, which means the combined
        tau = tau_1 * ... * tau_k cannot be reconstructed even if every OTHER
        participant reveals their secret. This is the 1-of-k trust assumption.
        """
        destroyed_val = self._tau  # save for display only
        self._tau = None
        self._destroyed = True
        return destroyed_val


class TrustedSetup:
    """
    Groth16 Trusted Setup Ceremony — orchestrates the multi-party process.

    Reference: https://www.rareskills.io/post/groth16

    Two phases:

    PHASE 1 — Powers of Tau (multi-party, circuit-independent):
      Multiple participants sequentially contribute their secret tau_i.
      Each participant updates the SRS and then destroys their secret.
      Result: EC points encoding [tau^0, tau^1, ..., tau^k] * G where
      tau = prod(tau_i), unknown to anyone.

    PHASE 2 — Circuit-Specific (single coordinator):
      Coordinator generates alpha, beta, gamma, delta (all secrets) and
      computes the circuit-specific SRS using Phase 1 output + QAP polys.
      The coordinator does NOT know tau — they compute everything via inner
      products with the Phase 1 points. They DO know alpha/beta/gamma/delta
      while setting up, then destroy all four.

      The public/private witness split is baked into Psi here:
        Psi_public_i  = [(beta*u_i(tau) + alpha*v_i(tau) + w_i(tau)) / gamma]_1
                        for i in [0, num_public)
        Psi_private_i = [(beta*u_i(tau) + alpha*v_i(tau) + w_i(tau)) / delta]_1
                        for i in [num_public, m)

      eta (H(tau)*t(tau) term) also divides by delta:
        eta_j = [tau^j * t(tau) / delta]_1  for j in [0, n-1)

    Published SRS (all EC points — no raw scalars):
      srs1, srs2, eta, [alpha]_1, [beta]_1, [beta]_2,
      [gamma]_2, [delta]_1, [delta]_2, psi_public, psi_private
    """

    def __init__(self, qap, num_public, participant_names=None):
        # UPGRADE: TrustedSetup now requires `num_public` so it can split the
        # Psi vector into public-side (divided by gamma) and private-side
        # (divided by delta) during Phase 2.
        self.qap = qap
        self.num_public = num_public  # ell: first `num_public` witness columns are public
        if participant_names is None:
            participant_names = ["Alice", "Bob", "Charlie"]
        self.participant_names = participant_names

    def run_ceremony(self):
        """Execute the full trusted setup and return the public SRS."""
        n = self.qap.n      # number of constraints
        m = self.qap.m      # number of columns
        ell = self.num_public

        assert 0 < ell <= m, f"num_public must be in (0, {m}], got {ell}"

        print(f"\n{'='*60}")
        print(f"  TRUSTED SETUP CEREMONY (FULL GROTH16)")
        print(f"{'='*60}")
        print(f"  Public inputs: witness[0:{ell}]  (shared with verifier)")
        print(f"  Private inputs: witness[{ell}:{m}]  (known only to prover)")

        # ============================================
        # PHASE 1: Powers of Tau (multi-party)
        # ============================================
        print(f"\n  PHASE 1: Powers of Tau (multi-party)")
        print(f"  {'-'*50}")
        print(f"  {len(self.participant_names)} participants will each contribute randomness.")
        print(f"  Security: as long as ONE participant deletes their secret,")
        print(f"  the combined tau is unknown to everyone.\n")

        # Need 2n-1 G1 powers (for eta computation up to tau^(2n-2)).
        # Need n G2 powers (for V(tau) evaluation by the prover).
        n_g1_powers = 2 * n - 1
        n_g2_powers = n

        powers_g1 = [G1] * n_g1_powers
        powers_g2 = [G2] * n_g2_powers

        participants = [CeremonyParticipant(name) for name in self.participant_names]

        for participant in participants:
            print(f"  >> {participant.name} enters the ceremony")
            print(f"     Generates secret tau_{participant.name}...")
            print(f"     Updating {n_g1_powers} G1 points and {n_g2_powers} G2 points...")

            powers_g1, powers_g2 = participant.contribute(powers_g1, powers_g2)

            destroyed = participant.destroy_secret()
            print(f"     DESTROYING secret tau_{participant.name} = {destroyed}  -->  None")
            print(f"     {participant.name} can never recover their secret.\n")

        tau_product_str = " * ".join(f"tau_{name}" for name in self.participant_names)
        print(f"  Phase 1 complete!")
        print(f"  Combined tau = {tau_product_str}")
        print(f"  Nobody knows this value (1-of-{len(self.participant_names)} trust assumption).")
        print(f"  Produced: {n_g1_powers} G1 powers, {n_g2_powers} G2 powers\n")

        # ============================================
        # PHASE 2: Circuit-Specific Setup (coordinator)
        # ============================================
        print(f"  PHASE 2: Circuit-Specific Setup (coordinator)")
        print(f"  {'-'*50}")
        print(f"  Coordinator generates alpha, beta, gamma, delta and computes")
        print(f"  circuit elements using Phase 1 EC points + QAP polynomials.")
        print(f"  The coordinator does NOT know tau.\n")

        # UPGRADE: full Groth16 needs FOUR circuit-level secrets (Part 1 only
        # had alpha, beta). gamma gates the public-input side of the SRS;
        # delta gates the private-input side plus the H(tau)*t(tau) term and
        # the prover's blinding. All four must be destroyed after setup.
        alpha = secrets.randbelow(p - 1) + 1
        beta  = secrets.randbelow(p - 1) + 1
        gamma = secrets.randbelow(p - 1) + 1   # UPGRADE: new
        delta = secrets.randbelow(p - 1) + 1   # UPGRADE: new
        gamma_inv = f_inv(gamma)               # UPGRADE: used to divide Psi_public
        delta_inv = f_inv(delta)               # UPGRADE: used to divide Psi_private and eta
        print(f"  Coordinator: Generated alpha = {alpha}")
        print(f"  Coordinator: Generated beta  = {beta}")
        print(f"  Coordinator: Generated gamma = {gamma}")
        print(f"  Coordinator: Generated delta = {delta}\n")

        # Raw tau powers the prover will use directly.
        srs1 = powers_g1[:n]
        srs2 = powers_g2[:n]

        # Scalar * generator elements
        # UPGRADE: [beta]_1, [gamma]_2, [delta]_1, [delta]_2 are all new in the
        # full protocol. The prover needs [beta]_1 and [delta]_1 to build
        # [B]_1 and the blinding terms in [C]_1; the verifier needs [gamma]_2
        # and [delta]_2 for the 4-pairing check.
        alpha_G1 = multiply(G1, alpha)
        beta_G1  = multiply(G1, beta)     # UPGRADE: needed by prover for [B]_1
        beta_G2  = multiply(G2, beta)
        gamma_G2 = multiply(G2, gamma)    # UPGRADE: verifier pairs against [X]_1
        delta_G1 = multiply(G1, delta)    # UPGRADE: prover's blinding and -rs*[delta]_1 term
        delta_G2 = multiply(G2, delta)    # UPGRADE: verifier pairs against [C]_1

        # ---- eta: [tau^j * t(tau) / delta]_1 for j = 0..n-2 ----
        # UPGRADE: divide each eta entry by delta. In Part 1 this was just
        # tau^j * t(tau) * G1. The /delta factor is what forces H(tau)*t(tau)
        # to always appear inside [C]_1 (whose pairing partner is [delta]_2),
        # so the factor cancels out in the final verification equation.
        # Compute tau^j * t(tau) * G1 via inner products with T_coeffs and
        # Phase 1 G1 powers (the coordinator never learns tau). Then scale
        # by delta_inv (the coordinator knows delta).
        print(f"  Coordinator: Computing eta ({n-1} G1 points)...")
        print(f"    eta_j = delta_inv * sum_k T_coeffs[k] * [tau^(j+k)]_1")
        T_coeffs = self.qap.T
        eta = []
        for j in range(n - 1):
            point = None
            for k in range(len(T_coeffs)):
                t_k = int(T_coeffs[k]) % p
                if t_k == 0:
                    continue
                term = multiply(powers_g1[j + k], t_k)
                point = term if point is None else add(point, term)
            # UPGRADE: Divide-by-delta (coordinator knows delta, multiplies by its inverse)
            if point is not None:
                point = multiply(point, delta_inv)
            eta.append(point)

        # ---- Psi_public and Psi_private ----
        # UPGRADE: Part 1 had a single `psi` list that the prover combined
        # with the ENTIRE witness inside [C]_1. That made the scheme not
        # zero-knowledge across public inputs — anyone seeing the proof and
        # the SRS could recompute Psi_i * a_i and check equalities.
        #
        # Full Groth16 SPLITS Psi in two:
        #   Psi_public_i  = C_i(tau) / gamma * G1   (public: i < ell)
        #   Psi_private_i = C_i(tau) / delta * G1   (private: i >= ell)
        # where C_i(tau) = w_i(tau) + alpha*v_i(tau) + beta*u_i(tau).
        # The /gamma and /delta factors cancel out against [gamma]_2 and
        # [delta]_2 at verification. Verifier uses Psi_public; prover uses
        # Psi_private.
        print(f"  Coordinator: Computing Psi ({ell} public + {m - ell} private G1 points)...")
        print(f"    C_i(tau) = w_i(tau) + alpha*v_i(tau) + beta*u_i(tau)")
        print(f"    Psi_public_i  = C_i(tau) * gamma_inv   (i < {ell})")
        print(f"    Psi_private_i = C_i(tau) * delta_inv   (i >= {ell})")
        psi_public = []
        psi_private = []
        for i in range(m):
            # Evaluate u_i, v_i, w_i at tau via inner products with srs1.
            w_c = list(self.qap.W_polys[i]) + [0] * max(0, n - len(self.qap.W_polys[i]))
            v_c = list(self.qap.V_polys[i]) + [0] * max(0, n - len(self.qap.V_polys[i]))
            u_c = list(self.qap.U_polys[i]) + [0] * max(0, n - len(self.qap.U_polys[i]))

            w_point = inner_product_ec(w_c[:n], srs1)   # w_i(tau) * G1
            v_point = inner_product_ec(v_c[:n], srs1)   # v_i(tau) * G1
            u_point = inner_product_ec(u_c[:n], srs1)   # u_i(tau) * G1

            alpha_v = multiply(v_point, alpha) if v_point is not None else None
            beta_u  = multiply(u_point, beta)  if u_point is not None else None

            combined = w_point
            if alpha_v is not None:
                combined = alpha_v if combined is None else add(combined, alpha_v)
            if beta_u is not None:
                combined = beta_u if combined is None else add(combined, beta_u)

            if i < ell:
                # UPGRADE: public columns get divided by gamma.
                scaled = multiply(combined, gamma_inv) if combined is not None else None
                psi_public.append(scaled)
            else:
                # UPGRADE: private columns get divided by delta.
                scaled = multiply(combined, delta_inv) if combined is not None else None
                psi_private.append(scaled)

        # ---- DESTROY circuit-specific toxic waste ----
        # UPGRADE: we now destroy gamma and delta in addition to alpha, beta.
        # Knowing any of these four would let an attacker forge proofs.
        print(f"\n  Coordinator: DESTROYING alpha, beta, gamma, delta (and their inverses)...")
        print(f"    alpha = {alpha}  -->  None")
        print(f"    beta  = {beta}   -->  None")
        print(f"    gamma = {gamma}  -->  None")
        print(f"    delta = {delta}  -->  None")
        alpha = None
        beta = None
        gamma = None
        delta = None
        gamma_inv = None
        delta_inv = None
        print(f"  Coordinator: all Phase-2 secrets destroyed.\n")

        # ---- Assemble final public SRS ----
        # UPGRADE: the SRS now carries four new EC elements plus a split Psi:
        #   Part 1 keys   -> {srs1, srs2, eta, alpha_G1, beta_G2, psi}
        #   Full Groth16  -> {srs1, srs2, eta, alpha_G1, beta_G1, beta_G2,
        #                     gamma_G2, delta_G1, delta_G2,
        #                     psi_public, psi_private, num_public}
        srs = {
            'srs1': srs1,
            'srs2': srs2,
            'eta': eta,
            'alpha_G1': alpha_G1,
            'beta_G1': beta_G1,              # UPGRADE
            'beta_G2': beta_G2,
            'gamma_G2': gamma_G2,            # UPGRADE
            'delta_G1': delta_G1,            # UPGRADE
            'delta_G2': delta_G2,            # UPGRADE
            'psi_public': psi_public,        # UPGRADE: was merged `psi` in Part 1
            'psi_private': psi_private,      # UPGRADE: was merged `psi` in Part 1
            'num_public': ell,               # UPGRADE: ell boundary
        }

        print(f"  {'='*50}")
        print(f"  CEREMONY COMPLETE — SRS Published")
        print(f"  {'='*50}")

        print(f"\n    srs1 ({n} G1 points) = [G1, tau*G1, ..., tau^{n-1}*G1]:")
        for i, pt in enumerate(srs1):
            print(f"      tau^{i}*G1 = {point_str(pt)}")

        print(f"\n    srs2 ({n} G2 points) = [G2, tau*G2, ..., tau^{n-1}*G2]:")
        for i, pt in enumerate(srs2):
            print(f"      tau^{i}*G2 = {point_str(pt)}")

        print(f"\n    eta ({n-1} G1 points) = [tau^j * t(tau) / delta * G1]:")
        for j, pt in enumerate(eta):
            print(f"      eta_{j} = {point_str(pt)}")

        print(f"\n    [alpha]1 = {point_str(alpha_G1)}")
        print(f"    [beta]1  = {point_str(beta_G1)}")
        print(f"    [beta]2  = {point_str(beta_G2)}")
        print(f"    [gamma]2 = {point_str(gamma_G2)}")
        print(f"    [delta]1 = {point_str(delta_G1)}")
        print(f"    [delta]2 = {point_str(delta_G2)}")

        print(f"\n    Psi_public ({len(psi_public)} G1 points, divided by gamma):")
        for i, pt in enumerate(psi_public):
            print(f"      Psi_public_{i} = {point_str(pt)}")

        print(f"\n    Psi_private ({len(psi_private)} G1 points, divided by delta):")
        for i, pt in enumerate(psi_private):
            print(f"      Psi_private_{i + ell} = {point_str(pt)}")

        print(f"\n  ALL toxic waste destroyed.")
        print(f"  Nobody knows tau, alpha, beta, gamma, or delta.")

        return srs


# ============================================================
#  PROVER
#
#  The Prover knows:
#    - The witness (private input satisfying the R1CS)
#    - The QAP polynomials (public, derived from the circuit)
#    - The SRS (public, from the trusted setup ceremony)
#
#  The Prover does NOT know:
#    - tau, alpha, beta (destroyed during ceremony)
#
#  The Prover evaluates polynomials at the secret tau by computing
#  inner products with the SRS vectors. This is the key trick:
#  you can evaluate a polynomial at a hidden point by linearly
#  combining the pre-computed [tau^i * G] points with your
#  polynomial's coefficients.
# ============================================================

class Prover:
    """
    Groth16 Prover — generates a proof of knowledge without revealing the witness.

    Reference: https://www.rareskills.io/post/groth16

    UPGRADE (vs Part 1): the prover now picks fresh random r, s for every proof.
    These scalars blind the proof so that two proofs of the same statement look
    independent (zero-knowledge + re-randomization). They are destroyed after use.

    Proof (3 EC points):
      [A]_1 = [alpha]_1 + [U(tau)]_1 + r*[delta]_1                  (G1)
      [B]_2 = [beta]_2  + [V(tau)]_2 + s*[delta]_2                  (G2)
      [C]_1 = sum_{i in private} a_i * Psi_private_i
              + <H, eta>
              + s*[A]_1 + r*[B]_1 - r*s*[delta]_1                   (G1)
    where [B]_1 is a prover-internal recomputation of B on G1:
      [B]_1 = [beta]_1 + [V(tau)]_1 + s*[delta]_1
    """

    def __init__(self, qap):
        self.qap = qap

    def generate_proof(self, srs):
        """Compute the Groth16 proof and return it for the Verifier."""
        n, m = self.qap.n, self.qap.m
        witness = self.qap.witness
        ell = srs['num_public']

        print(f"\n{'='*60}")
        print(f"  PROVER")
        print(f"{'='*60}")
        print(f"  Prover has the witness (private): {witness}")
        print(f"  Prover has the public SRS from the ceremony.")
        print(f"  Prover does NOT know tau, alpha, beta, gamma, or delta.\n")

        # UPGRADE: prover samples fresh blinding scalars r, s per proof.
        # These hide U(tau) and V(tau) behind random multiples of delta,
        # giving Groth16 its zero-knowledge property.
        r = secrets.randbelow(p - 1) + 1
        s = secrets.randbelow(p - 1) + 1
        print(f"  Prover: sampled fresh r = {r}")
        print(f"  Prover: sampled fresh s = {s}\n")

        # Pad polynomial coefficients to match SRS sizes
        U_coeffs = list(self.qap.U) + [0] * max(0, n - len(self.qap.U))
        V_coeffs = list(self.qap.V) + [0] * max(0, n - len(self.qap.V))
        H_coeffs = list(self.qap.H) + [0] * max(0, (n - 1) - len(self.qap.H))

        # ---- Step 1: [A]_1 = [alpha]_1 + [U(tau)]_1 + r*[delta]_1 ----
        # UPGRADE: added the r*[delta]_1 blinding term.
        print(f"  Step 1: [A]_1 = [alpha]_1 + [U(tau)]_1 + r*[delta]_1")
        U_eval_G1 = inner_product_ec(U_coeffs[:n], srs['srs1'])
        r_delta_G1 = multiply(srs['delta_G1'], r)
        A1 = add(add(srs['alpha_G1'], U_eval_G1), r_delta_G1)
        print(f"    [U(tau)]_1     = {point_str(U_eval_G1)}")
        print(f"    [alpha]_1      = {point_str(srs['alpha_G1'])}")
        print(f"    r*[delta]_1    = {point_str(r_delta_G1)}")
        print(f"    [A]_1          = {point_str(A1)}")

        # ---- Step 2: [B]_2 = [beta]_2 + [V(tau)]_2 + s*[delta]_2 ----
        # UPGRADE: added the s*[delta]_2 blinding term.
        print(f"\n  Step 2: [B]_2 = [beta]_2 + [V(tau)]_2 + s*[delta]_2")
        V_eval_G2 = inner_product_ec(V_coeffs[:n], srs['srs2'], identity=Z2)
        s_delta_G2 = multiply(srs['delta_G2'], s)
        B2 = add(add(srs['beta_G2'], V_eval_G2), s_delta_G2)
        print(f"    [V(tau)]_2     = {point_str(V_eval_G2)}")
        print(f"    [beta]_2       = {point_str(srs['beta_G2'])}")
        print(f"    s*[delta]_2    = {point_str(s_delta_G2)}")
        print(f"    [B]_2          = {point_str(B2)}")

        # ---- Step 3: [B]_1 (prover-internal) for use inside [C]_1 ----
        # UPGRADE: full Groth16 needs a G1 version of B for the r*[B]_1 term
        # in [C]_1. Same structure as [B]_2 but in G1.
        print(f"\n  Step 3: [B]_1 = [beta]_1 + [V(tau)]_1 + s*[delta]_1  (prover-internal)")
        V_eval_G1 = inner_product_ec(V_coeffs[:n], srs['srs1'])
        s_delta_G1 = multiply(srs['delta_G1'], s)
        B1 = add(add(srs['beta_G1'], V_eval_G1), s_delta_G1)
        print(f"    [B]_1          = {point_str(B1)}")

        # ---- Step 4: [C]_1 ----
        # UPGRADE: only the PRIVATE part of the witness goes into C.
        # The public part is re-built by the verifier (via Psi_public).
        # UPGRADE: added s*[A]_1 + r*[B]_1 - r*s*[delta]_1 blinding terms.
        print(f"\n  Step 4: [C]_1 = sum_priv a_i * Psi_private_i + <H,eta>")
        print(f"                   + s*[A]_1 + r*[B]_1 - r*s*[delta]_1")
        priv_scalars = [int(witness[i]) % p for i in range(ell, m)]
        C1_psi = inner_product_ec(priv_scalars, srs['psi_private'])
        C1_ht  = inner_product_ec(H_coeffs[:n - 1], srs['eta'])
        sA     = multiply(A1, s)
        rB     = multiply(B1, r)
        rs     = f_mul(r, s)
        neg_rs_delta = neg(multiply(srs['delta_G1'], rs))

        C1 = C1_psi
        C1 = C1_ht if C1 is None else add(C1, C1_ht)
        C1 = sA    if C1 is None else add(C1, sA)
        C1 = rB    if C1 is None else add(C1, rB)
        C1 = neg_rs_delta if C1 is None else add(C1, neg_rs_delta)

        print(f"    sum_priv a_i * Psi_private_i = {point_str(C1_psi)}")
        print(f"    <H, eta>       = [H(tau)*t(tau)/delta]_1 = {point_str(C1_ht)}")
        print(f"    s*[A]_1        = {point_str(sA)}")
        print(f"    r*[B]_1        = {point_str(rB)}")
        print(f"    -r*s*[delta]_1 = {point_str(neg_rs_delta)}")
        print(f"    [C]_1          = {point_str(C1)}")

        # UPGRADE: destroy r, s after use (one-time blinding).
        print(f"\n  Prover: DESTROYING r and s (one-time randomness)...")
        r = None
        s = None

        proof = {'A1': A1, 'B2': B2, 'C1': C1}

        print(f"\n  Final proof (3 EC points):")
        print(f"    [A]_1 = {point_str(A1)}")
        print(f"    [B]_2 = {point_str(B2)}")
        print(f"    [C]_1 = {point_str(C1)}")
        print(f"\n  >>> Prover sends proof = ([A]_1, [B]_2, [C]_1) to the Verifier >>>")

        return proof


# ============================================================
#  VERIFIER
#
#  The Verifier knows:
#    - The SRS (public, from the trusted setup ceremony)
#    - The proof (A1, B2, C1) received from the Prover
#
#  The Verifier does NOT know:
#    - The witness (the Prover's private input)
#    - tau, alpha, beta (destroyed during ceremony)
#
#  Verification is a single pairing equation:
#    e(-[A]1, [B]2) * e([alpha]1, [beta]2) * e([C]1, G2) = I_12
#
#  Why this works — expanding the proof elements:
#    [A]1 = (alpha + U(tau)) * G1
#    [B]2 = (beta  + V(tau)) * G2
#    [C]1 = (W(tau) + alpha*V(tau) + beta*U(tau) + H(tau)*t(tau)) * G1
#
#  The pairing equation reduces to:
#    U(tau)*V(tau) = W(tau) + H(tau)*t(tau)
#  which is the QAP equation at x = tau. Since the QAP holds
#  for ALL x (polynomial identity), it holds at tau too.
# ============================================================

class Verifier:
    """
    Groth16 Verifier — checks the proof without learning the private witness.

    Reference: https://www.rareskills.io/post/groth16

    UPGRADE (vs Part 1): the verifier now consumes the PUBLIC part of the
    witness (witness[:num_public]) to rebuild [X]_1 from Psi_public, and runs
    a 4-pairing equation instead of 3.

    Pairing check:
      e(-[A]_1, [B]_2) * e([alpha]_1, [beta]_2)
        * e([X]_1, [gamma]_2) * e([C]_1, [delta]_2) = 1
    where
      [X]_1 = sum_{i in public} a_i * Psi_public_i.

    Only needs the proof (3 EC points), the SRS, and the public inputs.
    """

    def verify(self, proof, srs, public_inputs):
        """
        Check the Groth16 pairing equation.

        If it holds, the Prover knew a valid witness whose public part matches
        `public_inputs` (with overwhelming probability under SDH assumptions).
        """
        A1 = proof['A1']
        B2 = proof['B2']
        C1 = proof['C1']
        ell = srs['num_public']
        assert len(public_inputs) == ell, \
            f"Verifier expected {ell} public inputs, got {len(public_inputs)}"

        print(f"\n{'='*60}")
        print(f"  VERIFIER")
        print(f"{'='*60}")
        print(f"  Verifier received proof = ([A]_1, [B]_2, [C]_1) from Prover.")
        print(f"  Verifier has the public SRS from the ceremony.")
        print(f"  Verifier has the public inputs: {list(public_inputs)}")
        print(f"  Verifier does NOT know the private witness, tau, alpha, beta, gamma, delta.\n")

        # UPGRADE: build [X]_1 from public inputs + Psi_public.
        # [X]_1 = sum_{i<ell} a_i * Psi_public_i
        #       = sum_{i<ell} a_i * [(beta*u_i(tau) + alpha*v_i(tau) + w_i(tau)) / gamma]_1
        # Pairing with [gamma]_2 cancels the gamma, recovering the public part
        # of W(tau) + alpha*V(tau) + beta*U(tau) in the exponent.
        print(f"  Step 1: [X]_1 = sum_{{i<{ell}}} a_i * Psi_public_i")
        pub_scalars = [int(public_inputs[i]) % p for i in range(ell)]
        X1 = inner_product_ec(pub_scalars, srs['psi_public'])
        print(f"    [X]_1 = {point_str(X1)}\n")

        print(f"  Step 2: check pairing equation")
        print(f"    e(-[A]_1, [B]_2) * e([alpha]_1, [beta]_2)")
        print(f"      * e([X]_1, [gamma]_2) * e([C]_1, [delta]_2) =? I_12\n")
        print(f"  Computing 4 pairings...")

        # --- About pairing and final_exponentiate ---
        #
        # See "Pairings for Beginners" by Craig Costello (in this folder):
        #   - Miller loop:          Chapter 5, Section 5.3 (p.75)
        #   - Final exponentiation: Chapter 7, Section 7.5 (p.113)
        #   - Background (Weil/Tate pairings): Sections 5.1-5.2 (pp.69-74)
        #
        # py_ecc pairing signature: pairing(G2_point, G1_point)
        # A production implementation would use raw Miller loops and a single
        # final_exponentiate for ~3x speedup.
        result = eq(
            FQ12.one(),
            final_exponentiate(
                pairing(B2, neg(A1)) *
                pairing(srs['beta_G2'], srs['alpha_G1']) *
                pairing(srs['gamma_G2'], X1) *
                pairing(srs['delta_G2'], C1)
            )
        )

        if result:
            print(f"\n  Result: PASS -- Proof is VALID!")
            print(f"  The Prover convinced the Verifier they know a valid witness")
            print(f"  matching the declared public inputs — WITHOUT revealing the")
            print(f"  private part. Zero knowledge achieved!")
        else:
            print(f"\n  Result: FAIL -- Proof is INVALID!")
            print(f"  The Prover could not demonstrate knowledge of a valid witness.")

        return result


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  GROTH16 (FULL)")
    print("  RareSkills Homework 12 (Week 13)")
    print("  Ref: https://www.rareskills.io/post/groth16")
    print("=" * 60)

    # ---- Step 1: Choose R1CS ----
    print("\n  Select R1CS:")
    print("    1. Default (out = 3x^2*y + 5xy - x - 2y + 3, x=100, y=100)")
    print("    2. Custom (enter your own R1CS matrices)")
    r1cs_choice = input("  Choice [1/2]: ").strip()

    if r1cs_choice == "2":
        A, B, C, witness = get_custom_r1cs()
    else:
        A, B, C, witness = get_default_r1cs()
        print(f"\n  Using default R1CS: out = 3x^2*y + 5xy - x - 2y + 3")
        print(f"  x = 100, y = 100, out = {witness[1]}")

    n_constraints = A.shape[0]
    m_columns = A.shape[1]
    print(f"\n  R1CS dimensions: {n_constraints} constraints x {m_columns} variables")
    print(f"  Witness: {witness}")

    # Validate R1CS
    print(f"\n  Validating R1CS constraints:")
    assert validate_r1cs(A, B, C, witness), "R1CS validation FAILED!"
    print("  R1CS is valid!\n")

    # ---- Step 2: UPGRADE — choose public/private witness split ----
    # Groth16 separates "public inputs" (seen by verifier) from "private
    # inputs" (known only to prover). The first `num_public` witness columns
    # are public. For the default circuit, [1, out] are the natural publics.
    print("  Select public-input count (first k witness columns are public):")
    print(f"    Default: 2  (witness[0]=1 constant and witness[1]=out)")
    pub_raw = input(f"  num_public [default 2, max {m_columns}]: ").strip()
    num_public = int(pub_raw) if pub_raw else 2
    assert 0 < num_public <= m_columns, f"num_public must be in (0, {m_columns}]"
    print(f"  Public inputs (verifier sees):  witness[0:{num_public}] = "
          f"{list(witness[:num_public])}")
    print(f"  Private inputs (prover only):   witness[{num_public}:] = "
          f"{list(witness[num_public:])}")

    # ---- Step 3: Choose Lagrange mode ----
    print("\n  Select Lagrange interpolation mode:")
    print("    1. Manual (from scratch)")
    print("    2. Library (galois)")
    lag_choice = input("  Choice [1/2]: ").strip()

    mode = "library" if lag_choice == "2" else "manual"

    # ---- Step 4: Build QAP and run Groth16 protocol ----
    print(f"\n{'='*60}")
    print(f"  QAP CONSTRUCTION (Lagrange: {mode})")
    print(f"{'='*60}")

    qap = QAP(A, B, C, witness, lagrange_mode=mode)

    print(f"  n (constraints/gates): {qap.n}")
    print(f"  m (columns/wires):     {qap.m}")
    print(f"  deg(U) = {poly_degree(qap.U)},  deg(V) = {poly_degree(qap.V)}")
    print(f"  deg(W) = {poly_degree(qap.W)},  deg(H) = {poly_degree(qap.H)}")
    print(f"  deg(T) = {poly_degree(qap.T)}")

    qap.print_polynomials()

    assert qap.verify(), "QAP polynomial verification FAILED!"
    print(f"\n  QAP VERIFIED: U(x)*V(x) = W(x) + H(x)*T(x)")

    # ---- Trusted Setup Ceremony ----
    # UPGRADE: pass num_public so the ceremony can split Psi into
    # psi_public (divided by gamma) and psi_private (divided by delta).
    ceremony = TrustedSetup(
        qap,
        num_public=num_public,
        participant_names=["Alice", "Bob", "Charlie"],
    )
    srs = ceremony.run_ceremony()

    # ---- Prover generates proof ----
    prover = Prover(qap)
    proof = prover.generate_proof(srs)

    # ---- Verifier checks proof ----
    # UPGRADE: verifier consumes only the public slice of the witness,
    # not the whole thing. This is what makes Groth16 non-trivial — the
    # verifier must be able to check the proof without the private inputs.
    verifier = Verifier()
    public_inputs = witness[:num_public]
    result = verifier.verify(proof, srs, public_inputs)

    if result:
        print(f"\n  *** GROTH16 PROOF VERIFIED SUCCESSFULLY ({mode} Lagrange) ***")
    else:
        print(f"\n  *** GROTH16 PROOF VERIFICATION FAILED ({mode} Lagrange) ***")


if __name__ == "__main__":
    main()

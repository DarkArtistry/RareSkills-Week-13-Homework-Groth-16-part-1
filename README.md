# Groth16 (FULL) — Week 13 Homework

From-scratch Python implementation of the complete Groth16 zkSNARK, built on
top of the Part 1 scaffolding from Week 12.

**Reference:** <https://www.rareskills.io/post/groth16> (formulas at the end of the article)

## What was upgraded vs Part 1

Every code site that implements a new piece of the protocol is tagged with an
`UPGRADE: ...` comment explaining the change. Grep for them:

```bash
grep -n "UPGRADE" groth16.py
```

| Added  | Role |
| ------ | ---- |
| `gamma` (γ) | Gates the public-input side of the SRS. `Psi_public_i = (β·u_i(τ) + α·v_i(τ) + w_i(τ))/γ · G1`. Pairing with `[γ]₂` cancels the division at verification. |
| `delta` (δ) | Gates the private-input side, the `H(τ)·t(τ)` term, and the prover's blinding. Pairing with `[δ]₂` cancels the division. |
| `r`, `s` | Prover's fresh per-proof randomness. Blinds `A`, `B`, and `C` so two proofs of the same statement look independent (zero knowledge + re-randomization). |
| Public/private split | First `num_public` witness columns are public (verifier side, via `Psi_public`); the rest are private (prover side, via `Psi_private`). |

## Protocol

### Trusted setup (single ceremony, reusable)

Phase 1 (multi-party Powers of Tau) is unchanged from Part 1: each participant
contributes a secret `tau_i`, updates the SRS, and destroys their secret. The
combined `tau = ∏ tau_i` is unknown as long as at least ONE participant was honest.

Phase 2 (coordinator) now generates **four** secrets and destroys all of them:

- `[α]₁, [β]₁, [β]₂` — added `[β]₁` for the prover's `[B]₁`.
- `[γ]₂` — verifier only.
- `[δ]₁, [δ]₂` — prover (blinding) + verifier.
- `[τⁱ]₁` for `i ∈ [0, n)`, `[τⁱ]₂` for `i ∈ [0, n)`.
- `ηⱼ = [τʲ · t(τ) / δ]₁` for `j ∈ [0, n-1)`.
- `Psi_public_i` for `i < ℓ` (divided by γ).
- `Psi_private_i` for `i ≥ ℓ` (divided by δ).

### Prover

Samples fresh `r, s ← F_p` per proof. Computes

```
[A]₁ = [α]₁ + [U(τ)]₁ + r·[δ]₁
[B]₂ = [β]₂ + [V(τ)]₂ + s·[δ]₂
[B]₁ = [β]₁ + [V(τ)]₁ + s·[δ]₁            (internal, needed for [C]₁)
[C]₁ = Σ_{i∈priv} aᵢ·Psi_private_i
       + ⟨H, η⟩
       + s·[A]₁ + r·[B]₁ − r·s·[δ]₁
```

Then destroys `r, s` and sends `([A]₁, [B]₂, [C]₁)`.

### Verifier

Rebuilds the public side from declared public inputs:

```
[X]₁ = Σ_{i<ℓ} aᵢ · Psi_public_i
```

Checks

```
e(−[A]₁, [B]₂) · e([α]₁, [β]₂) · e([X]₁, [γ]₂) · e([C]₁, [δ]₂) = 1 ∈ GT
```

This is the full 4-pairing Groth16 equation.

## Usage

```bash
python groth16.py
```

Prompts for:

1. R1CS choice — default (`out = 3x²y + 5xy − x − 2y + 3`, `x = y = 100`) or custom.
2. `num_public` — defaults to `2` (first two witness columns = `[1, out]`).
3. Lagrange mode — from-scratch or `galois` library.

## Files

- `groth16.py` — full implementation (R1CS → QAP → Setup → Prover → Verifier).
- `f1039491-…_RareSkils_Homework_11.pdf` — homework 11 (Part 1) brief.
- `PairingsForBeginners - Craig Costello.pdf` — pairing theory reference; see
  Ch. 5 §5.3 (Miller loop) and Ch. 7 §7.5 (final exponentiation).

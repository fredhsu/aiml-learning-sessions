---
created: 2026-06-27T00:21:55Z
id: 38257caf-f2e9-45c1-b04f-530c7d07d9e1
tags:
- reading
- convolution
- inductive-bias
- equivariance
title: Week 12 · S3 · Read — What Inductive Bias Convolution Encodes
---

# Week 12 Session 3 — What Inductive Bias Convolution Encodes

## The bias, in my own words

The readings for this session focus on taking convolutions from just matrix
operations to an inductive bias that acts as a prior. This is built upon a few
properties and assumptions of convolutions often attributed to image data:

1. **Assumed locality of data** — Using the filters provides a local view of the
   input, such as the fact that the pixels in a local area of an image are
   locally relevant. *(Mechanism: small filters.)*

2. **Translation equivariance** — Equivariance means $f(\text{shift}_s\,x) =
   \text{shift}_s(f(x))$; detecting then shifting is the same as shifting then
   detecting. This means that we can move the input and the response moves with
   it — same feature but new location. This does not hold generally; it doesn't
   work for rotation / scaling. The mechanism for providing this is **weight
   sharing**.

3. **Invariance provided by pooling** — this makes features relevant regardless
   of location, keeping *local* invariance. This holds with the argument that we
   detect-first then discard the location information. *(This detect-first /
   discard-later ordering is my own framing — it is not stated explicitly in the
   reading.)*

4. **Hierarchical features that benefit from depth/stacking**, such as
   edges → parts → objects. The receptive field here grows as $1 + L(k-1)$.
   *(Property: compositionality/hierarchy in the data; abstraction is the
   network's response to it.)*

## Where equivariance does the work, and how invariance is recovered

A bare conv layer is **translation-equivariant**: shift the input and the
response tracks the feature to its new location — position is *preserved*, not
discarded. **Pooling** is what introduces invariance, and only *local*
invariance: a max-pool over a window is unchanged under shifts small enough to
keep the feature inside the window, but a large shift moves the feature out and
the invariance breaks. Global shift-invariance is built up gradually as pooling
stacks shrink the spatial map across depth. The ordering matters: detection is
inherently positional (the filter slides across locations), so you must detect
*first* and discard position *second* — reversing it would throw away the
spatial structure before any filter could read it.

## The honest limit (bad-prior example)

This leads to some limitations, based on the assumption that the location is not
a relevant feature. An example here would be a weather map where finding a storm
has a different interpretation based on where it was found for the people in a
given location. If a storm is detected in Kansas, it has a different meaning for
those in Kansas than for someone in California.

*Mechanism of the failure: the feature looks identical everywhere (a storm is a
storm), so conv detects and even locates it fine — but weight sharing forces the
layer to interpret it identically everywhere. The label depends on absolute
position, and a position-blind shared filter cannot attach "matters here /
doesn't matter there" without being fed absolute coordinates as extra input.*

## Updated conjecture answers (vs. pre-reading bets)

- **Three good-prior properties.** Pre-reading list was muddled (two of three
  weren't data properties). Corrected: locality → filters; translation
  equivariance/stationarity → weight sharing; compositionality/hierarchy →
  depth. Replaced "abstraction" (network-side) with "compositionality"
  (data-side).
- **Bad-prior bet.** Pre-reading: "location on a map." Sharpened to the
  label-depends-on-position mechanism above; cleaner alternative on hand is the
  axis-is-absolute case (registered medical images, audio spectrograms across
  frequency).
- **Equivariance vs. invariance.** Pre-reading swapped them. Corrected from the
  index derivation: conv gives *translation* equivariance (output shifts with
  input), pooling recovers *local* invariance. GBC caveat caught on second pass:
  equivariant to translation only, **not** rotation/scale.

## Calibration

- **Boundary/overlap, third instance.** Receptive field for two stacked 3×3
  layers: predicted 9×9, actual 5×5 — adjacent receptive fields overlap by
  $k-1$, they don't sit end-to-end. Rule: $1 + L(k-1)$. Pre-check flagged for
  Week 13 (attention connectivity = "how far can information move in one step,"
  the opposite answer).
- **Map-to-vocabulary slip.** Index arithmetic for the shift was correct both
  times, but the *word* attached to it flipped (equivariance ↔ invariance).
  Different failure class than the arithmetic one. Fix that worked: derive the
  equation first, then read the term off the equation — never reach for the word
  from memory.

## Links

- [[Week11_S1_ImplicitBias_Math]] — bias from the *algorithm*; this week is bias
  from the *structure* (the rowspace/nullspace decomposition reused on a
  structured operator).
- [[Week12_S1_Conv_Math]] — the Toeplitz form, output-size formula, and four
  fundamental subspaces (null(K) = the filter's blind spot).
- **Forward → Week 13 (Attention):** convolution *hard-codes* its connectivity
  (a fixed band); attention *learns* it (content-dependent mixing). Both are
  linear maps from values to outputs; they differ in how the mixing weights are
  set. The receptive-field "one step" question above is where the contrast bites.

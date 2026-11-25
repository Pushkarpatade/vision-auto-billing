# VisionCheckout: Production Readiness Roadmap

## Current State

This repository is a demonstration, not a production checkout system. The
current detector can locate broad object categories such as bottles, cans, and
some coloured packages. It then infers a brand, pack size, and price from image
colour or fixed rules. Those inferred values must not be used to create a
customer charge.

The current product catalogue is also an in-memory Python dictionary. It is
reset when the service restarts, has no store or SKU ownership, and is not a
source of truth for live retail prices or tax rules.

## Product Principle

Separate these two outcomes at every stage:

1. **Object detection** answers: "there is an item at this location."
2. **Product identification** answers: "this exact store SKU and pack variant
   is present, with enough evidence to bill it."

Only verified product identification may add an item to a payable basket. A
low-confidence or ambiguous item must be marked `REVIEW_REQUIRED`, shown to
the customer/operator, and never silently assigned a price.

## First Retail Scope

Start with one controlled checkout lane and 100--300 packaged SKUs. Exclude
loose produce, medicines, alcohol, reflective packs, transparent packs, and
stacked items until each has its own validated workflow. A constrained first
scope is how the recognition model, cameras, and operating process can be
measured accurately.

## Required Architecture

```
Camera frame(s)
  -> item detection and multi-frame tracking
  -> crop quality checks (blur, glare, occlusion, view coverage)
  -> SKU recognition (trained classifier + OCR + visual embeddings)
  -> catalogue match and confidence calibration
  -> billable / review-required decision
  -> immutable basket event and POS payment
```

The vision system should propose a SKU. The store product master must supply
the SKU code, tax class, active price, promotion eligibility, size, and stock
policy. Vision must never be the price authority.

## Milestone 1: Truthful Checkout Contract

Build this before improving the model:

- Replace inferred prices with `ProductMatch` evidence: SKU, confidence,
  model/version, OCR evidence, and match status.
- Add statuses: `BILLABLE`, `REVIEW_REQUIRED`, `UNKNOWN`, and `OCCLUDED`.
- Require a configured confidence threshold and margin over the second-best SKU
  before billing automatically.
- Store the catalogue in a database and associate every SKU with a store,
  currency, tax code, pack size, effective price period, and active state.
- Log each decision, image reference, model version, catalogue version, and any
  operator override for audit and model improvement.
- Make the UI clearly show "needs confirmation" rather than showing a guessed
  item as verified.

**Acceptance gate:** run a labelled test set containing the intended SKUs,
different pack sizes, lighting conditions, rotations, damaged packs, partial
views, and similar-looking products. The lane may auto-bill only if precision
on billed items meets the agreed business threshold. Unknown or ambiguous items
must be routed to review, not misbilled.

## Milestone 2: Data and Recognition

- Create a SKU dataset with 200--500 images per SKU across real lane cameras,
  orientations, distances, lighting, and packaging revisions.
- Train product-level recognition. A generic COCO YOLO model is useful for
  locating items but cannot identify retail SKUs or pack sizes.
- Combine three signals: trained SKU classifier/embedding retrieval, package
  OCR, and geometry/pack-size checks. Calibrate their confidence with held-out
  store data.
- Use 2--3 fixed cameras or a guided turntable/tray. Multiple views reduce
  hidden labels and item overlap substantially.
- Track objects across several frames and decide only when evidence is stable.

## Milestone 3: Checkout and Operations

- Integrate with the retailer's POS/ERP catalogue and inventory APIs; cache
  safely for outages and reconcile later.
- Use the POS as the payment/tax authority and create idempotent basket and
  payment events.
- Add an operator review screen for unknown, duplicate, and occluded items.
- Add security controls: authenticated staff actions, per-store configuration,
  encryption, retention policy, monitoring, and audit logs.
- Validate loss prevention using video sequences and transaction reconciliation.
  Pairwise box overlap alone is not evidence of theft.

## Measurable Production Metrics

- SKU identification precision and recall, reported per SKU and pack size.
- Auto-bill precision: incorrect charges divided by all automatic charges.
- Unknown/review rate and operator correction rate.
- Duplicate-item and missed-item rate per transaction.
- End-to-end decision latency and lane throughput.
- Performance drift after product packaging changes.

## Immediate Next Implementation

Implement a persistent product master and a fail-closed matching API. The
existing UI should receive a recognition status for every object and only total
items whose match status is `BILLABLE`. This preserves the current demo while
making it safe to evolve into a real checkout workflow.

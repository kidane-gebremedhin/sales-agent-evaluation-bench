# Synthesis memo — Pushkarna, Zaldivar, Kjartansson, *Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI* (arXiv 2204.01075, FAccT 2022)

**Reader:** Kidane (TRP1 Week 11) · **Date:** 2026-04-29 · **Memo length:** ~1 page.

## What the paper argues

Pushkarna et al. (Google Research) propose **Data Cards** as a
structured, user-centric documentation layer that sits one rung above
Gebru's *Datasheets for Datasets* in the responsible-AI documentation
stack. The central insight is that documentation must be **layered for
multiple stakeholders**: a hurried product manager needs a one-line
"what is this", a lead engineer needs a one-paragraph "should I use
this", and a downstream auditor needs full provenance. The paper's
contribution is the **telescopic / periscopic / microscopic**
framework — three depths of detail that a single artifact serves
simultaneously, plus a recommended set of "essential facts" sections
across the dataset lifecycle (origins, collection, annotation,
intended use, quality, gaps).

## What I took from it for Tenacious-Bench

The [`datasheet.md`](../datasheet.md) for Tenacious-Bench v0.1 is
explicitly built on the layered framework:

1. **Telescopic — one line at the top.** "A 237-task, machine-
   verifiable evaluation bench for B2B engineering-talent sales
   agents…" — readable in a single breath. This is the line that
   determines whether anyone reads further; Pushkarna correctly calls
   it the *highest-leverage sentence in the document*.

2. **Periscopic — one paragraph per Gebru section.** Each numbered
   section (Motivation, Composition, Collection, Preprocessing, Uses,
   Distribution, Maintenance) opens with a *Periscopic* block —
   roughly the depth a lead engineer needs to decide whether to
   adopt the bench, without yet committing to read 50 lines of
   per-field schema.

3. **Microscopic — per-field documentation in §2.** The full
   `task_id` / `primary_dimension` / `rubric` / `source_provenance`
   field-by-field block is the depth a re-implementer or an auditor
   needs. It lives below the periscopic block deliberately, so a
   reader who only needs the periscopic answer is not slowed by the
   microscopic detail.

## Where I disagree

Pushkarna's framework argues for **one Data Card per dataset**. For
Tenacious-Bench this is the wrong shape, and the disagreement is
substantive:

The dataset has **two natural audiences with conflicting needs**:

- **The Tenacious executive (CEO / CFO).** Cares about: "is the
  trained component safe to deploy?" This audience needs the kill-
  switch trigger, the public-signal lossiness in `signal_grounding`,
  and the inter-rater caveats. Most of Pushkarna's Data Card
  structure is irrelevant — they will not read field-level provenance.

- **The HuggingFace Hub user (open-source ML practitioner).** Cares
  about: "can I reproduce the headline number?" This audience needs
  the schema, the rubric predicate registry, and the loader
  quickstart. Most of the executive-relevant content (Tenacious's
  ACV math, brand-tail risk) is irrelevant.

A single Data Card optimized for either reader degrades for the
other. The paper acknowledges multi-stakeholder documentation as the
*motivation* for the layered structure but then folds everything back
into a single artifact, asking readers to drill in or zoom out.

**My response in this repo:** the documentation is split across
`datasheet.md` (HF user, Pushkarna-conformant) and the eventual
`memo/memo.pdf` (Tenacious executive, two-page format). They share
provenance (the same `evidence_graph.json`) but have different
foregrounded content. Pushkarna's framework explicitly does not
contemplate this split — its working assumption is that one document
can serve all readers if the layering is good enough.

**Concrete cost of accepting Pushkarna's one-card framing:** the
microscopic per-field block in §2 of the datasheet would have
expanded from 12 lines to ~80 if I had attempted to embed the
executive-decision content. A reader looking for the rubric predicate
registry would have to skim past two paragraphs of ACV math.

## Open question for Act V publication

Pushkarna does not address **machine-readable** Data Cards. The
HuggingFace dataset card format is a constrained markdown subset; the
schema's `rubric_dimensions` and `task_object_schema` blocks are
already JSON Schema. v0.2 of Tenacious-Bench should publish a
machine-readable card (JSON) alongside the human card (markdown), so
that automated tooling can index the bench without re-parsing prose.
This is the next-step direction Pushkarna's paper points toward but
does not concretize.

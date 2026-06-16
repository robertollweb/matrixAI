# PR2 Runbook Template — MatrixAI Use Cases

This template defines the six-section format shared by all PR2 use case runbooks. Every case (current and future) must follow this structure to be consistent for technical evaluators.

---

## Format

### Section 1 — The problem

*1–2 paragraphs. Starts from an industry pain, not from a system capability.*

Rules:
- Name the industry and the role of the person who has the problem.
- State the problem in operational terms: what breaks, what cannot be proven, what takes too long.
- End with a concrete question the system must answer (e.g., "which exact model version produced this decision?").
- Do not mention MatrixAI capabilities in this section.

### Section 2 — The solution

*3–5 bullet points or short paragraphs. Describes what MatrixAI does, without jargon.*

Rules:
- Each bullet maps to one pain stated in Section 1.
- Name the specific mechanism (registry, composite hash, dry-run, HMAC signature) but explain what it does, not how it works internally.
- End with one sentence on the reproducibility guarantee.

### Section 3 — Run it yourself

*Command + expected output (literal, truncated to ~25 lines).*

Rules:
- Single command from the project root. Works on macOS, Linux, and Windows PowerShell.
- No external dependencies, no API keys, data included.
- Expected output is literal (copy-pasted from the actual run), with long hashes truncated to 8–16 chars + `…`.
- If data is auto-generated, note it: "Dataset included (auto-generated on first run)."

### Section 4 — The result

*Two sub-sections: model metric + value metric.*

**Model metric** — a table with:
- Component | Role | Accuracy (or other ML metric)
- Baseline (majority class or trivial rule)
- Honest note on expected real-world performance

**Value metric** — 3–5 bullets stating what the system can now answer that it could not before. Expressed in operational terms (time saved, risk reduced, compliance demonstrated), not in accuracy numbers.

### Section 5 — Limits

*Bullet list. Honest and specific. Required for regulated domains (clinical, financial).*

Rules:
- Start with the most important limit, especially regulatory disclaimers for clinical/financial cases.
- Each limit states what the system does NOT do and optionally what a better alternative would be.
- Include dataset limits (synthetic, controlled vocabulary, distribution assumptions).
- Include technical limits (model class, missing features, hardening gaps).

### Section 6 — What is free and what is paid

*Table with two columns: Layer | Status*

Rules:
- Status is either `**Core — free**` or `Paid tier` (no other values).
- "Core — free" items are what attracts users: training, local registry, basic inference, audit log.
- "Paid tier" items are what sustains the business: managed infrastructure, SLA, compliance reports, integrations, advanced features.
- The boundary must be consistent across all four cases (see Monetization coherence below).

---

## Header metadata

Every runbook starts with:

```markdown
# Case N — Title

> **Español:** [path to Spanish version]

**Industry:** Name
**Difficulty:** Beginner | Intermediate | Advanced
**Runtime:** ~N seconds
```

---

## Monetization coherence across all four cases

The free/paid boundary is the same in all four cases. This is intentional: a buyer comparing cases should see a consistent model, not four ad-hoc decisions.

| What is always free (Core) | What is always paid |
|---|---|
| Training any model locally | Managed registry with retention + access control |
| Registering models with signed `entry_hash` | Production API with SLA |
| Local registry with tamper detection | Compliance/audit report generation |
| Basic inference and composite pipelines | Enterprise integrations (SMTP, HL7/FHIR, etc.) |
| Decision audit log per decision | Advanced model features (SHAP, continual learning, HTTP serving) |
| Dry-run simulation, HMAC signatures, rollback | Human-in-the-loop workflows and support |

**Pattern:** the system is free to use locally. The paid tier is operating it at production scale with the reliability, retention, and compliance guarantees that regulated industries require.

---

## Cross-case index

| Case | File (EN) | File (ES) | Industry | Central capability |
|---|---|---|---|---|
| Case 1 — Credit Scoring | [CREDIT_SCORING.md](cases/CREDIT_SCORING.md) | [CREDIT_SCORING.md](../es/cases/CREDIT_SCORING.md) | Financial / Fintech | Registry P21 + audit trail |
| Case 2 — Text Routing | [TEXT_ROUTING.md](cases/TEXT_ROUTING.md) | [TEXT_ROUTING.md](../es/cases/TEXT_ROUTING.md) | SaaS / Customer Support | Composite model P21 + embeddings P19 |
| Case 3 — Clinical Risk | [CLINICAL_RISK.md](cases/CLINICAL_RISK.md) | [CLINICAL_RISK.md](../es/cases/CLINICAL_RISK.md) | Healthcare | Classification P4 + linear attribution |
| Case 4 — Agent Alert | [AGENT_ALERT.md](cases/AGENT_ALERT.md) | [AGENT_ALERT.md](../es/cases/AGENT_ALERT.md) | IT Operations | Real action P20 + signed trace |

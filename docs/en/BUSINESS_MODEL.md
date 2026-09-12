# MatrixAI Business Model

MatrixAI is **two bodies of software, with two licenses and one author**.

| | License | Price |
|---|---|---|
| **MatrixAI core and CLI** (`matrixai-core`) | GNU AGPL v3 (`AGPL-3.0-only`) | free |
| **MatrixAI Studio** (this product) | proprietary — see `LICENSE` | paid |

## The core is free software, and stays free

The core language, runtime, CLI, training pipeline, registry, HTTP serving,
deployment tools and documentation are published under the GNU Affero General
Public License version 3. The license verification record is in
[LICENSE_VERIFICATION.md](LICENSE_VERIFICATION.md).

That license lets anyone use, study, modify and redistribute the core under its
terms — including **building their own studio on top of it**, without asking
permission and without paying anything. That is a real alternative to buying
this product, and it is written here on purpose.

## MatrixAI Studio is a commercial product

MatrixAI Studio — the interface, the Studio backend, the web and the licensing
API — is **not** free software. It is covered by the proprietary terms in
`LICENSE`, which ships in the root of this package alongside `TERCEROS.md`.

**The Studio exists today, and it is what you are holding.** Until 2026-09-12
this document described it as planned *after* the v1.0 roadmap, "only if real
adoption shows demand", and expected to talk to the core "over HTTP instead of
importing Python modules directly". Both statements have been overtaken by the
product and this section replaces them: the Studio is built, it is sold, and
its backend imports the core's Python modules directly.

That direct import is not a conflict with the AGPL. The AGPL is strong
copyleft: a work that links with AGPL code and is then distributed has to be
distributed under the AGPL — *by anyone who is not the copyright holder*. The
core and the Studio have the same single copyright holder, who can publish his
own work as free software for everyone **and** use it under different terms in
his own product. No third party holds rights in the core that this could
infringe.

## What you receive when you buy the Studio

- **The Studio**, under the proprietary license in `LICENSE`.
- **The core, under AGPL-3.0**, with its full license text and with every right
  that license grants you over it: use it, study it, modify it, redistribute it.
  The Studio's license does not restrict that and does not try to.
- **Third-party libraries**, each one keeping its own terms.

`TERCEROS.md` lists all three, next to `LICENSE` in the root of the package.
What is sold here is this Studio, not the right to use the core: nobody needs
to buy that.

## Donations

Voluntary donations may be accepted through GitHub Sponsors or an equivalent
platform to help cover hosting costs and development time on the free core.
Donations are not required to use it.

## Brand

MatrixAI, matrixaistudio.org, and the public project identity belong to the
project author. External providers are mentioned only when technically
necessary to describe configurable compatibility.

## Limits of this document

This is a summary of the author's licensing position, not the license itself.
The governing texts are `LICENSE` for the Studio and the AGPL v3 text that
ships with the core; where this summary and those texts differ, **the texts
win**.

It is not legal advice, and it does not answer what the AGPL means *inside your
organisation*. Whether you may embed, extend, or redistribute something that
carries an AGPL core is a question for your own counsel — worth asking before
you deploy rather than after.

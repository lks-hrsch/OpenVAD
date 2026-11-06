# 0027: Project licence

Status: Accepted (2026-11-06)

## Context

The repository carried no licence, and the README planned a permissive one.
The owner's requirements, in order: commercial use must be allowed, modifications should be made public, and the owner's name should be referenced.

Apache 2.0 and European Union Public Licence 1.2 were argued against the Mozilla Public License 2.0; the GNU General Public License 3.0, its Affero variant and `MIT-0` were reviewed afterwards.

- Apache 2.0 allows closed forks, so it gives up public modifications.
- European Union Public Licence 1.2 may reach hosted services, but that reading is untested, a merge with Mozilla-licensed code probably escapes it, and it has no cure after termination.
- The GNU General Public License 3.0 and its Affero variant license the whole work when it is distributed, which stops closed products from shipping the engine.
- `MIT-0` requires neither publication nor attribution.

## Decision

Everything in this repository is licensed under the Mozilla Public License 2.0 (SPDX `MPL-2.0`): code, trained weights and documents.

- `LICENSE` at the repository root holds the unedited text from Mozilla.
- Every Rust and Python source file in `inferencing/` and `training/` starts with the owner's copyright line and the Exhibit A notice. New files do too.
- Other files rely on `LICENSE`, as Exhibit A allows.
- Crate and package manifests declare `MPL-2.0`.
- For the trained weights, the safetensors file is the Source Code Form (section 1.13); training data is not part of it, and [0024](0024-training-data-licence-policy.md) forbids redistributing it.
- Third-party material keeps its own licence: the `docs/tudcd-scr` submodule, and the datasets and third-party weights that the scripts fetch.

## Consequences

- Commercial users may link the engine into closed products; their own separate files stay private (sections 1.7 and 3.3).
- Anyone distributing changed engine files must publish those files (sections 3.1 and 3.2).
- Server-side use triggers no publication duty, so a voicebot that runs a private fork publishes nothing.
- New logic in new files stays private, even in distributed forks.
- The owner's name is protected in source files only (section 3.4); recipients of a binary are told only where to get the source (section 3.2).
- Distributed fine-tuned weights stay under this licence (sections 1.10 and 3.1), as far as weights are protected by copyright at all.
- This licence does not settle whether trained weights count as adapted material under CC BY-NC-SA; that question stays open in [0024](0024-training-data-licence-policy.md) and is answered before release.

## Open questions

- Whether trained weights are protected by copyright, which decides whether the licence conditions bind them.
- Whether TU Dresden or the company hold any rights in the code, the weights or the thesis, and whether university rules limit how the thesis is licensed. This has not been checked.
- How the wheels include `LICENSE`, since the root file may not reach the member packages. Settle this before the first wheel is published.

# Changelog

This file records user-facing changes to the standalone FiggieJAX Engine.
Only changes to this independent package belong in these release notes.

## 0.16.1 - 2026-11-18

- Add project attribution to the citation metadata.
- Document how to prepare a clean source snapshot for public release.
- Include the GPU and league diagrams in the local preview instructions.

## 0.16.0 - 2026-11-17

- Add an educational full-JAX episodic REINFORCE learner with private linear
  policy heads, batched complete-round collection, gradients, and repeated SGD
  updates compiled together. Keep the existing host example for comparison.
- Capture every-seat rewards, assign learner seats per round, and exclude
  truncations from targets. Use separate training and evaluation key domains.
- Add Docker GPU tests, explicit backend checks, held-out evaluation, source
  hashes, synchronized timing, and local parameter artifacts. Add no dependencies.

## 0.15.4 - 2026-11-17

- Add cloning, uv installation, runtime and development setup, consumer-project
  installation, namespace conflict guidance, and explicit platform scope.
- Explain parallel learning collection, private actor routing, every-seat reward
  capture, truncation, and live-round retention without adding a trainer.
- Add repository, event-flow, and local-validation Mermaid overviews and preview commands.
- Rebuild wheels from source archives and verify isolated consumer installation.
- Include the Dockerfile in validation-image source archives so local Docker
  validation is reproducible from the archive.
- Reject private artifacts in distributions and extend local ignore rules.
- Exclude generated results from Ruff checks in archives and Docker images
  that do not contain Git metadata.
- Settle Gymnasium and PettingZoo deadlines before offering another live action;
  preserve every-seat terminal rewards and the reset-time Gym handoff.
- Remove a shared conversation link from the rules guide.
- Preserve simulator rules, public adapter signatures, and existing dependency pins.

## 0.15.3 - 2026-11-17

- Document NVIDIA GPU setup, parallel environment batches, and measured throughput tuning.
- Add a self-contained integration guide for GPU-resident learner state, frozen
  opponents, scripted agents, and host control boundaries.
- Add a README Mermaid overview for a modular, application-owned league.
- Keep setup instructions independent and the engine free of trainer dependencies.

## 0.15.2 - 2026-11-10

- Use GitHub-compatible Markdown math delimiters in the documentation.
- Define equation notation and reward-scale units explicitly.
- Correct the explanation of the example baseline-update order.

## 0.15.1 - 2026-11-10

- Explain the compiled JAX simulation workflow directly in the README.
- Add runnable batch instructions and Mermaid execution and training diagrams.
- Clarify the host boundaries of the examples and optional adapters.
- Remove GitHub Actions automation and keep all validation local.

## 0.15.0 - 2026-11-10

### Added

- Optional Gymnasium, PettingZoo, and OpenSpiel adapters for the native engine.
- A complete-round rollout example for four- and five-player games.
- A seeded, masked linear REINFORCE example with held-out evaluation seeds.
- Guides for RL training, private observations, and finite action IDs.
- Standalone contribution and citation metadata.
- A release validator for source checks, example smoke runs, and isolated wheel
  and source archive checks.

### Changed

- Clarified the rules guide to describe only standalone engine behavior and to
  identify upstream audit results as historical evidence.

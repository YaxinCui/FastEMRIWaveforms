# Project edit lock

<!-- 2026-09-01 18:24 CST (mac): Record the initial active editor and branch. -->

<!-- 2026-09-01 19:24 CST (mac): Mark the validated Mac handoff as prepared while retaining the lock until the user confirms the host switch. -->

<!-- 2026-09-01 19:27 CST (mac): Release Mac editing after the validated implementation and data commit reached GitHub. -->

<!-- 2026-09-01 19:31 CST (linux): Acquire the project edit lock after the user-directed Ubuntu pull; begin CUDA-side consistency validation. -->

<!-- 2026-09-01 20:04 CST (linux): Mark CPU/CUDA dual-host validation complete while retaining the lock until the user directs synchronization or a host switch. -->

- Branch: `codex/apple-silicon-dual-host`
- Active editor: Ubuntu CUDA collaborator
- Host: Ubuntu x86_64, Intel Core i5-13400F, NVIDIA GeForce RTX 2080 Ti
- Lock acquired: 2026-09-01 19:31 CST (Asia/Shanghai)
- Previous lock: Apple M3 Pro/macOS released at 2026-09-01 19:27 CST
- State: Ubuntu CPU and CUDA comparisons plus both fast suites pass. Two
  explicit CUDA host/device boundary fixes and their regression tests are
  recorded in `collaboration/linux/HANDOFF.md`; lock retained pending user
  direction.

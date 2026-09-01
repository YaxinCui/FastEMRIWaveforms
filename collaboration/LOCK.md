# Project edit lock

<!-- 2026-09-01 18:24 CST (mac): Record the initial active editor and branch. -->

<!-- 2026-09-01 19:24 CST (mac): Mark the validated Mac handoff as prepared while retaining the lock until the user confirms the host switch. -->

<!-- 2026-09-01 19:27 CST (mac): Release Mac editing after the validated implementation and data commit reached GitHub. -->

<!-- 2026-09-01 19:31 CST (linux): Acquire the project edit lock after the user-directed Ubuntu pull; begin CUDA-side consistency validation. -->

<!-- 2026-09-01 20:04 CST (linux): Mark CPU/CUDA dual-host validation complete while retaining the lock until the user directs synchronization or a host switch. -->

<!-- 2026-09-01 20:10 CST (linux): Release Ubuntu editing after committing successful CPU/CUDA validation; Mac may acquire only after the user completes push and pull synchronization. -->

<!-- 2026-09-01 20:28 CST (mac): Acquire the edit lock after fast-forwarding to the completed Ubuntu handoff for Apple regression verification. -->

<!-- 2026-09-01 20:38 CST (mac): Release editing after the final Apple rebuild accepted Ubuntu's CPU/CUDA validation fixes. -->

<!-- 2026-09-01 20:38 CST (mac): Bind the released final state to the completed Mac acceptance commit before remote synchronization. -->

- Branch: `codex/apple-silicon-dual-host`
- Active editor: none; dual-host adaptation accepted
- Host: Apple M3 Pro, macOS 26.5.2, arm64
- Lock acquired: 2026-09-01 20:28 CST (Asia/Shanghai)
- Lock released: 2026-09-01 20:38 CST (Asia/Shanghai)
- Previous lock: Ubuntu/RTX 2080 Ti released at 2026-09-01 20:10 CST
- State: Mac post-CUDA verification passed at pulled handoff `5526b336`; final
  acceptance is recorded in commit `b1dce6b2`. No host holds the edit lock.

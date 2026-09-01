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

<!-- 2026-09-01 21:06 CST (linux): Temporarily acquire the edit lock only to document the verified high-memory Kerr table and its user-confirmed transfer to Mac. -->

<!-- 2026-09-01 21:06 CST (linux): Release the documentation-only lock so Mac can pull, verify the transferred table, and begin high-memory Kerr acceptance. -->

<!-- 2026-09-01 21:13 CST (mac): Acquire the edit lock after verifying the transferred 5.09 GB Kerr table for high-memory Apple acceptance. -->

<!-- 2026-09-01 21:29 CST (mac): Release editing after full-table Kerr amplitude/waveform acceptance and preparation of the small Ubuntu comparison artifact. -->

- Branch: `codex/apple-silicon-dual-host`
- Active editor: `none`
- Last editor: `mac` — Apple M3 Pro, macOS 26.5.2, arm64, 36 GiB unified memory
- Lock released: 2026-09-01 21:29 CST (Asia/Shanghai)
- Previous lock: Mac acquired at 2026-09-01 21:13 CST
- State: Mac high-memory Kerr reference is ready. Ubuntu may acquire only
  after the user synchronizes and pulls this branch; until then both hosts are
  read-only.

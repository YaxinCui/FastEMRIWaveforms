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

<!-- 2026-09-01 21:33 CST (linux): Acquire the edit lock after pulling Mac commit f7028a39 to run full-table Kerr CPU/CUDA consistency acceptance. -->

<!-- 2026-09-01 21:36 CST (linux): Mark both full-table comparisons successful while retaining the lock until the generated reports and handoff are synchronized. -->

<!-- 2026-09-01 21:38 CST (linux): Release editing after successful full-table CPU/CUDA acceptance and preparation of the user-directed report commit. -->

<!-- 2026-09-01 22:21 CST (mac): Acquire the project edit lock for the user-directed deep Apple GPU feasibility study; only isolated Metal prototypes and documented measurements may change until release. -->

<!-- 2026-09-01 22:57 CST (mac): Release the edit lock after isolating, measuring, documenting, and regression-checking the native-Metal feasibility study; changes remain local until a user-directed commit/push. -->

<!-- 2026-09-01 23:14 CST (mac): Reacquire the project edit lock for the user-directed strict-precision Metal mode-summation exploration; changes remain confined to the Mac PoC, research notes, lock, and handoff. -->

<!-- 2026-09-01 23:28 CST (mac): Release editing after the strict full-chain double-single Metal sum passed the one-year engineering gate, retained useful acceleration, and was documented and regression-checked; changes remain local until a user-directed commit/push. -->

<!-- 2026-09-01 23:30 CST (mac): Acquire the project edit lock solely for the user-directed Metal research handoff, commit, and push on the existing collaboration branch. -->

<!-- 2026-09-01 23:31 CST (mac): Release editing after the reviewed handoff set was finalized for the user-directed commit and push; the commit itself records this released state. -->

<!-- 2026-09-01 23:14 CST (linux): Acquire the edit lock to record the user-requested FP64/FP32/FP16 mixed-precision research plan and validation gates. -->

<!-- 2026-09-01 23:16 CST (linux): Mark the mixed-precision documentation complete while retaining the lock pending user-directed synchronization. -->

<!-- 2026-09-01 23:36 CST (linux): Reconcile the independently recorded lock histories after pulling Mac commit 5c872d31, preserve both hosts' work, and confirm Linux ownership for its pending documentation sync. -->

<!-- 2026-09-01 23:45 CST (linux): Release editing after documenting the mixed-precision plan and the exact missing strict-Metal artifact required from Mac. -->

<!-- 2026-09-01 23:51 CST (mac): Acquire the project edit lock after pulling Ubuntu commit 0120e06c to generate the requested strict-Metal waveform artifact/report and synchronize the existing branch. -->

<!-- 2026-09-01 23:59 CST (mac): Release editing after the five-case strict-Metal artifact/report passed integrity and regression checks and was finalized for the user-directed handoff commit/push. -->

<!-- 2026-09-02 00:04 CST (linux): Acquire the edit lock after pulling Mac commit fcfac79d to validate all five strict-Metal arrays against Linux CPU/CUDA and LPA-weighted metrics. -->

<!-- 2026-09-02 00:19 CST (linux): Mark strict-Metal CPU/CUDA and LPA-weighted validation complete while retaining the lock for user-directed review or synchronization. -->

<!-- 2026-09-02 00:36 CST (linux): Release editing after committing the passed strict-Metal validation as 1564eaee for the user-directed GitHub handoff. -->

<!-- 2026-09-02 10:53 CST (mac): Acquire the project edit lock after confirming a clean, synchronized codex/apple-silicon-dual-host worktree; isolate the remaining cross-host discrepancy by freezing the exact strict-Metal summation ABI inputs and adding a portable CPU/CUDA kernel validator. -->

<!-- 2026-09-02 11:04 CST (mac): Release editing after the frozen five-case summation inputs, Mac CPU kernel replay, integrity checks, documentation, and Ubuntu replay contract all passed; changes remain local pending a user-directed commit/push. -->

<!-- 2026-09-02 11:27 CST (mac): Reacquire the project edit lock solely for the user-directed frozen-summation handoff, commit, and push after fetching the existing branch and confirming zero local/remote divergence. -->

<!-- 2026-09-02 11:28 CST (mac): Release editing after the user-directed synchronization record and final pre-commit audit passed; this released state is included in the handoff commit. -->

<!-- 2026-09-02 11:41 CST (linux): Acquire the edit lock after fast-forwarding to Mac commit b74fb40e and verifying the frozen-input handoff identities. -->

<!-- 2026-09-02 13:32 CST (linux): Mark frozen CPU/CUDA kernel replay complete after correcting the portable validator's mixed CUDA pointer placement; retain the lock pending user-directed synchronization. -->

- Branch: `codex/apple-silicon-dual-host`
- Active editor: `linux`
- Host: Ubuntu x86_64 / NVIDIA GeForce RTX 2080 Ti
- Lock acquired: 2026-09-02 11:41 CST (Asia/Shanghai)
- Previous lock: Mac released at 2026-09-02 11:28 CST
- State: Mac strict Metal, Ubuntu CPU, and Ubuntu CUDA frozen-input kernel
  replays all pass the binding gates. Ubuntu retains the lock while the
  validator fix, two Linux reports, and completed handoff await review or a
  user-directed commit/push.

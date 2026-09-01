# Dual-host collaboration

<!-- 2026-09-01 18:24 CST (mac): Bootstrap the shared coordination area for the Apple Silicon/CUDA adaptation. -->

This directory coordinates work on `codex/apple-silicon-dual-host` between the
Apple Silicon development host and the Ubuntu CUDA validation host.

## Ownership

- `mac/` is writable only by the macOS collaborator. The Ubuntu collaborator
  may read it but must not modify it.
- `linux/` is writable only by the Ubuntu collaborator. The macOS collaborator
  may read it but must not modify it after the bootstrap commit.
- `LOCK.md` and this file are shared coordination files. Only the collaborator
  holding the project edit lock may modify shared project files.

Every handoff entry and source change made for this adaptation must include an
Asia/Shanghai timestamp, the host (`mac` or `linux`), and its purpose. Generated
or binary artifacts that cannot contain comments must be described in the
owning host's handoff log.

Before switching hosts, the active collaborator updates its handoff log and
`LOCK.md`, commits all required source and large-file pointers/data, pushes the
existing branch to GitHub, and asks the user to pull it on the other host. No
additional branch may be created.

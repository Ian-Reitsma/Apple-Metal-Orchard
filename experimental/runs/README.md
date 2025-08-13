# Runs

Store logs and outputs from experimental runs here, organised by experiment
phase or tag. The directory is ignored by Git to keep transient data out of the
repository.

## Next Steps
Add subdirectories or scripts as needed to track specific experiments. Include a README in each subdirectory describing the purpose, commit hash, and any runtime flags such as `USE_FLASH_ATTN`. Remove the directory when the experimental path is retired.

## Contributor Protocol
- Follow `../../AGENTS.md` for repository rules even though this directory is ignored by Git.
- Do not add large artifacts or commit generated run outputs; keep this directory untracked except for README files that explain experiment context.
- Record experiment details using inline code for commands and flags, avoiding fenced code blocks.
- Before submitting related changes elsewhere, run `cmake -S . -B build -G Ninja` and `cmake --build build --target test` from the repository root and note any failures in the pull request.
- Use `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` when configuring offline so tests link against the trimmed `third_party/googletest` tree or a system package.
- Keep commits focused with a single imperative summary line and reference changed files by path and line number in the pull request message.

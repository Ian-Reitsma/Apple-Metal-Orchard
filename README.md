# Apple Metal Orchard

This repository experiments with FlashAttention kernels for PyTorch's Metal (MPS) backend.

## Building the custom PyTorch wheel

A helper script is provided under `scripts/build_custom_torch.sh`. It installs the
required build tools, builds a wheel from the included `pytorch` submodule and then
installs torchvision and torchaudio from source.

```
./scripts/build_custom_torch.sh
```

If `torchaudio` fails with `ModuleNotFoundError: No module named 'cmake'`, ensure
that the `cmake` and `ninja` Python packages are installed before running the
script.

To reduce clone size when setting up a new environment you can fetch the repo and
its submodules with shallow history:

```
git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/Ian-Reitsma/Apple-Metal-Orchard.git
```

## Running tests

Tests rely on PyTorch. After installing the wheel, run:

```
pytest
```
## Working with submodule changes

The repository tracks exact commits of its submodules. Any commit in a submodule
must exist on the remote **before** the root repository updates its pointer.
Follow this sequence when editing `submodules/metal-tensor`:

```bash
cd submodules/metal-tensor
# edit files, then commit
git commit -am "Describe change"
git push origin agent/codex

cd ../..
# update root to the new submodule commit
git add submodules/metal-tensor
git commit -m "Update metal-tensor submodule pointer"
git push origin agent/codex
```

When responding to additional requests after pushing, synchronise both
repositories with:

```bash
git pull --recurse-submodules
git submodule update --init --recursive
```

If the submodule commit is not pushed first, later clones will fail during
`git submodule update` because the referenced commit cannot be fetched.

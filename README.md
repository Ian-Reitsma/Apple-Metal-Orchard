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
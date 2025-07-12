#pragma once
#include <ATen/ATen.h>
at::Tensor orchard_flash_attn_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float scale,
    bool causal);

# GPU Baseline Results

## ref_mac1-1-100-gpu_bs2_no_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac1-1-100-gpu_bs2_no_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 690.5580424668518, "tokenize_s": 0.2837, "copy_s": 3.3001, "copy_mb_s": 0.25, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721076791360974, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": false}, "metal_ver": "32023.620", "seed": 42, "timeline": [], "avg_gpu_w": 5495.266666666666, "avg_gpu_temp": null}
```

## ref_mac1-2-100-gpu_bs2_no_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac1-2-100-gpu_bs2_no_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 711.4782379289438, "tokenize_s": 0.2749, "copy_s": 3.0979, "copy_mb_s": 0.26, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721076791360974, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": false}, "metal_ver": "32023.620", "seed": 42, "timeline": [], "avg_gpu_w": 5589.197278911564, "avg_gpu_temp": null}
```

## ref_mac1-3-100-gpu_bs2_no_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac1-3-100-gpu_bs2_no_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 705.8344630591638, "tokenize_s": 0.2646, "copy_s": 3.0466, "copy_mb_s": 0.27, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721076791360974, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": false}, "metal_ver": "32023.620", "seed": 42, "timeline": [], "avg_gpu_w": 5558.649659863946, "avg_gpu_temp": null}
```

## ref_mac1-1-100-gpu_bs2_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac1-1-100-gpu_bs2_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 707.160807838668, "tokenize_s": 0.303, "copy_s": 2.9254, "copy_mb_s": 0.28, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721076791360974, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": true, "sha": "e036b69162f349edcc4f5d14b31e837ec3ce67738ff188c228efeeaa9ea7cb2e", "built": "2025-07-01T01:52:10", "metal": "32023.620"}, "metal_ver": "32023.620", "seed": 42, "timeline": [], "avg_gpu_w": 5477.044217687075, "avg_gpu_temp": null}
```

## ref_mac1-2-100-gpu_bs2_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac1-2-100-gpu_bs2_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 703.6690537356124, "tokenize_s": 0.303, "copy_s": 2.8627, "copy_mb_s": 0.29, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721076791360974, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": true, "sha": "e036b69162f349edcc4f5d14b31e837ec3ce67738ff188c228efeeaa9ea7cb2e", "built": "2025-07-01T01:52:10", "metal": "32023.620"}, "metal_ver": "32023.620", "seed": 42, "timeline": [], "avg_gpu_w": 5523.006802721088, "avg_gpu_temp": null}
```

## ref_mac1-3-100-gpu_bs2_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac1-3-100-gpu_bs2_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 704.2607577203288, "tokenize_s": 0.2831, "copy_s": 3.001, "copy_mb_s": 0.27, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721076791360974, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": true, "sha": "e036b69162f349edcc4f5d14b31e837ec3ce67738ff188c228efeeaa9ea7cb2e", "built": "2025-07-01T01:52:10", "metal": "32023.620"}, "metal_ver": "32023.620", "seed": 42, "timeline": [], "avg_gpu_w": 5385.847972972973, "avg_gpu_temp": null}
```

## ref_mac2-1-100-gpu_bs2_no_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac2-1-100-gpu_bs2_no_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 688.1733145788575, "tokenize_s": 0.3451, "copy_s": 2.2403, "copy_mb_s": 0.37, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721955718286335, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": false}, "metal_ver": "unknown", "seed": 42, "timeline": [], "avg_gpu_w": 5246.579470198675, "avg_gpu_temp": null}
```

## ref_mac2-2-100-gpu_bs2_no_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac2-2-100-gpu_bs2_no_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 676.5519950399839, "tokenize_s": 0.3676, "copy_s": 2.246, "copy_mb_s": 0.36, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721955718286335, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": false}, "metal_ver": "unknown", "seed": 42, "timeline": [], "avg_gpu_w": 5297.5, "avg_gpu_temp": null}
```

## ref_mac2-3-100-gpu_bs2_no_flash_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_mac2-3-100-gpu_bs2_no_flash_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 698.5418441940792, "tokenize_s": 0.4703, "copy_s": 2.5004, "copy_mb_s": 0.33, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721955718286335, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": false}, "metal_ver": "unknown", "seed": 42, "timeline": [], "avg_gpu_w": 5509.074324324324, "avg_gpu_temp": null}
```

## ref_flash_enabled_mac2-1-100_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_flash_enabled_mac2-1-100_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 687.1595177259235, "tokenize_s": 0.3505, "copy_s": 2.3515, "copy_mb_s": 0.35, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721955718286335, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": true, "sha": "e036b69162f349edcc4f5d14b31e837ec3ce67738ff188c228efeeaa9ea7cb2e", "built": "2025-07-01T02:55:39", "metal": "unknown"}, "metal_ver": "unknown", "seed": 42, "timeline": [], "avg_gpu_w": 5395.248344370861, "avg_gpu_temp": null}
```

## ref_flash_enabled_mac2-2-100_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_flash_enabled_mac2-2-100_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 682.99557505822, "tokenize_s": 0.4647, "copy_s": 2.2624, "copy_mb_s": 0.36, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721955718286335, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": true, "sha": "e036b69162f349edcc4f5d14b31e837ec3ce67738ff188c228efeeaa9ea7cb2e", "built": "2025-07-01T02:55:39", "metal": "unknown"}, "metal_ver": "unknown", "seed": 42, "timeline": [], "avg_gpu_w": 5400.075657894737, "avg_gpu_temp": null}
```

## ref_flash_enabled_mac2-3-100_bf16
```json
{"schema": "orchard_bench_v1", "tag": "ref_flash_enabled_mac2-3-100_bf16", "commit": "d512501", "dataset_sha": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92", "batch": 2, "seq": 512, "micro_steps": 100, "total_tokens": 102400, "throughput_tps": 656.9517577408882, "tokenize_s": 0.6787, "copy_s": 2.2831, "copy_mb_s": 0.36, "train_s": 0.0, "host_to_gpu_mb": 0.82, "avg_loss_stub": 0.8721955718286335, "max_grad": 159.0, "dtype": "torch.bfloat16", "bf16": true, "flash": {"enabled": true, "sha": "e036b69162f349edcc4f5d14b31e837ec3ce67738ff188c228efeeaa9ea7cb2e", "built": "2025-07-01T02:55:39", "metal": "unknown"}, "metal_ver": "unknown", "seed": 42, "timeline": [], "avg_gpu_w": 5279.06329113924, "avg_gpu_temp": null}
```


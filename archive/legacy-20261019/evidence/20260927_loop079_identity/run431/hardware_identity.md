# Run431 — installed 910B3 hardware identity versus maximum-rate certificate

Read-only host inventory, 2026-09-27. No service or NPU workload launched. Raw `npu-smi`, DMI and SHA records are saved alongside this file.

## Joined identity

- `npu-smi info -m`: eight Ascend **910B3** devices.
- Rank0 board: **IT21HMDC_Bin6**, Huawei, board ID 0x62, PCI device 0xD802, 64 GiB HBM. The other seven board rows were already frozen in Run421's eight-device preflight.
- Rank0 common: 20 AI Core, configured `Aicore Freq=1800 MHz`, idle `curFreq=800 MHz`. The configured rate is not an attested maximum under boost/error policy; idle current clock cannot estimate loaded peak.
- DMI: manufacturer **WUZHOU**, product **S900K3**, OEM SKU field left generic. `npu-smi -t product` is unsupported by this installed driver. HBM clock query reports 1600 MHz but does not certify usable bandwidth or overclock/error envelope.

The [S900K3 manufacturer product page](https://wuzhoucloud.com/pt/storage1327.html) calls it an eight-Ascend-module server with HCCS full mesh and advertises 392 GB/s bidirectional interconnect. It does not identify this host's exact 910B3 compute bin, BF16/W4A8 maximum operation rate, applicable peak-clock/error envelope or measured cut bandwidth. The manufacturer identity strengthens platform provenance but still does **not** bind the different [Huawei Atlas 800T A2 white-paper](https://e.huawei.com/cn/documents/products/computing/b0b253c4d5da4e05ba80a39e3dc65fd2) 313/376 TFLOPS FP16 options to this Wuzhou S900K3 board.

An older [manufacturer S900K3 launch article](https://wuzhoucloud.com/news/index397.html) advertises 2.56 PFLOPS FP16 for the whole 4U system but describes interconnect as four-NPU full-mesh islands at 60 GB/s bidirectional, unlike the current product page's eight-way/392 GB/s wording. No exact board/bin or revision join is given. Record these as competing *product-family marketing specifications*, not a capacity or topology certificate for the measured board.

## Bound decision

Run423's strict `W⁻/C⁺` theorem remains correct. Run431 improves board/host identity but supplies neither a work-matched genuine aggregate `C⁺` nor a universal necessary `W⁻`. No numeric Hardware/Resource or Product TPS ceiling is promoted. Treat the OEM HCCS figure as a design advertisement, not the attainable/full-graph communication lower-bound denominator. An isolated GEMM or HCCL test would give attained service, not the missing no-faster-than capacity certificate.

Next hardware inquiry: obtain official maximum issue/rate and clock/bin applicability for this exact board or bound all permitted compute engines from a documented microarchitecture and maximum clock. Separately keep practical mixed-load attainable compute/HBM/HCCL measurement for an Engineering Bound. Do not substitute a vendor's unrelated FP16 option for W4A8 operations.

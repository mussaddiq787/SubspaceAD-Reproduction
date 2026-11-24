# SubspaceAD Reproduction: Lab Notes

## Environment (18 Oct 2026)
- Code: SubspaceAD @ commit ef56d5c
- Python 3.10.22 (conda env: subspacead, Miniforge)
- PyTorch 2.14.1+cu130, torchvision 0.29.1
- NVIDIA driver 595.84
- GPU: Quadro RTX 3000 (6 GB, sm_75)
- OS: Linux Mint 22.3

## Log
- 18 Oct: Env created, PyTorch installed, GPU check passed (sm_75 supported).
- 18 Oct: MVTec-AD from HF mirror micguida1/mvtech_anomaly_detection (official mydrive links 404). Verified 15 cats, 3629 train / 1725 test = official counts.
- 18 Oct: FULL REPRO k=1 (large@448, aug30, seed42): I-AUROC 93.57 | P-AUROC 97.21 | AU-PRO 93.24 (paper giant@672: 97.1 / 97.5). Worst: screw .70, cable .78, transistor .84 (aug auto-disabled in code). Runtime ~18 min.
- 19 Nov: k-shot ablation (large@448, seed42). I-AUROC k1/k2/k4 = 93.57/94.36/95.21; P-AUROC 97.21/97.38/97.58; AU-PRO 93.24/93.53/94.09. Transistor .842->.914->.927 (data scarcity confirmed). Screw .700->.711->.692 (more data does NOT help; pixel AUROC stays .986 -> image-score aggregation issue).

## 19 Nov 2026 — Four-category Qwen input preparation
- Ran the requested k=1, DINOv2-large@448, aug30, pca_ev0.99, seed42 command on bottle/cable/screw/transistor on branch qwen-explain.
- Verified 493 per-image score rows and 493 readable, matching overlays (179 normal / 314 defective). Per-category counts: 83 / 150 / 160 / 100.
- Image AUROC: bottle 1.0000, cable 0.7777, screw 0.7122, transistor 0.8221. The first two match the earlier full 15-category run; screw/transistor do not (earlier 0.6997/0.8417).
- Saved logs confirm changed normal exemplars: screw 282.png -> 194.png; transistor 204.png -> 044.png. The authors seed once globally and shuffle sequentially, so changing the category list changes the random streams. Do not describe the latter two differences as rounding or modify results to match. No author sampling/scoring code was changed.
- Outputs: results/qwen_run/ (one nested run directory); checks/diagnosis: results/qwen_run/input_verification.json. F1 remains N/A without a validation threshold.
- explain_qwen.py was absent from the project and no matching script was found in Downloads; Qwen evaluation and grounded/ungrounded comparisons have NOT run.

## 19 Nov 2026 — Full replay and Qwen grounded/ungrounded evaluation
- Full 15-category replay with overlays verified: 1,725 scores/overlays; all saved metrics and exemplars match original. Target Image AUROC: bottle 1.0000, cable 0.7777, screw 0.6997, transistor 0.8417.
- Attached explain_qwen.py copied to project root. accelerate 1.15.0 installed; FP16 Qwen pilot succeeded with no OOM, quantization, API or parsing fixes. Pilot archived separately.
- Fixed final sample: same 120 IDs in both conditions; 100 defective / 20 normal; five per folder type.

| Measure | Grounded | Ungrounded |
| --- | ---: | ---: |
| Detector accuracy (120 samples) | 87.5% | 87.5% |
| Defect-type accuracy (100 defective samples; uniform chance 20.0%) | 19.0% | 17.0% |
| bottle: defect type (n=15; chance 33.3%) | 33.3% | 20.0% |
| cable: defect type (n=40; chance 12.5%) | 12.5% | 7.5% |
| screw: defect type (n=25; chance 20.0%) | 20.0% | 32.0% |
| transistor: defect type (n=20; chance 25.0%) | 20.0% | 15.0% |
| Normal images assigned none (20 samples) | 55.0% | 50.0% |
| Unparseable JSON (120 answers) | 0.0% | 0.0% |
| Invalid output schema/vocabulary | 0.0% | 0.0% |

- Oracle test-label F1 thresholds and prescribed NORMAL -> none responses make these exploratory explanation results. Single seed/small sample/2B model; taxonomy supplied; no reference-based validation of explanation wording, location or severity.
- Category-list sensitivity changes both selected exemplars and random augmentation streams; do not claim an isolated exemplar effect or quantify publication uncertainty from two runs.
- Detailed report: reports/qwen_evaluation_results.md; raw CSVs/comparison/examples under results/qwen_run_full/.

## 2026-11-21 — Original-image crop v2

Task 1 used saved v1 answers only: among 20 reference-normal images, the detector said NORMAL for 11 and Qwen answered none for all 11; it said DEFECTIVE for nine and Qwen answered none for zero. The 55% v1 normal-image none rate therefore came entirely from detector false positives, not ignoring the NORMAL instruction.

Crop mode adds a second original-pixel view. Its ROI is the bounding box of any changed RGB pixels between the saved overlay and the original resized with exactly the renderer's PIL resize; a 15% per-side margin is clipped to bounds and mapped back to native coordinates. No reference mask or per-image reference label determines this ROI. Empty-mask fallback is explicit. Default overlay mode, prompts, thresholds, sampling and greedy decoding remain unchanged; v1 source/output hashes and eight focused checks were verified.

| Measure | v1 grounded | v1 ungrounded | v2 crop grounded |
| --- | ---: | ---: | ---: |
| Detector accuracy (120 samples) | 87.5% | 87.5% | 87.5% |
| Defect-type accuracy (100 defective; chance 20%) | 19.0% | 17.0% | 22.0% |
| Bottle (n=15; chance 33.3%) | 33.3% | 20.0% | 46.7% |
| Cable (n=40; chance 12.5%) | 12.5% | 7.5% | 12.5% |
| Screw (n=25; chance 20%) | 20.0% | 32.0% | 20.0% |
| Transistor (n=20; chance 25%) | 20.0% | 15.0% | 25.0% |
| Normal images assigned none (n=20) | 55.0% | 50.0% | 45.0% |
| Unparseable JSON | 0.0% | 0.0% | 0.0% |
| Invalid output schema/vocabulary | 0.0% | 0.0% | 0.0% |

GPU: Quadro RTX 3000; torch.float16; full view 448x448 plus crop 448x448. Peak PyTorch allocated 4.092 GiB, reserved 4.137 GiB. These include model loading and generation, not all driver/display allocations. No OOM, 336 fallback or quantization was used.

Crop geometry limitation: 73/120 padded boxes covered the entire image; median area fraction 1.0, mean 0.9129. Thus most samples had little or no actual zoom. Empty-difference fallbacks: zero. The ROI represents the displayed, morphologically denoised overlay region, not a raw anomaly-map argmax.

v2 instruction adherence: 2 of the 11 reference-normal images with a NORMAL detector decision received a defect type. Hence the decrease from 55% to 45% is two new instruction violations, while the nine detector false positives remain unchanged. Across all 17 NORMAL decisions, including reference-defective images, v2 violated the none instruction four times.

Interpretation: the 22% versus 19% defect-type difference is three additional correct labels among 100 defective samples, not evidence of reliable fine-grained explanations. Both input representation (including image count) and wording changed; this does not isolate zoom benefit. Same oracle test-set thresholds, small fixed sample, one seed, known taxonomy and 2B model limitations still apply. Locations, wording and severity have no independent language-reference validation.

Three deterministic examples (first correct type, wrong type and reference-normal/none row; type correctness does not validate the text):

- **correct_type** — bottle/broken_large -> `broken_large`; detector `DEFECTIVE`. The bottle has a large, visible crack in the upper portion of the opening, which appears to be a structural failure.
- **wrong_type** — bottle/broken_large -> `contamination`; detector `DEFECTIVE`. A white, crystalline residue is visible on the inner surface of the bottle's rim, indicating a contamination.
- **normal_assigned_none** — bottle/good -> `none`; detector `NORMAL`. The image shows a bottle from a top-down perspective, with no visible defects such as cracks, contamination, or structural damage.

# SubspaceAD + Qwen explanation results — 2026-11-19

Full detector replay: 1,725 score rows and readable overlays, all 15 selected exemplars and all saved metric columns match the original reproduction at saved precision.

Qwen3-VL-2B-Instruct; float16 on Quadro RTX 3000; no quantization, API fixes, parser changes or prompt tuning. Original attached script used unchanged. Bottle pilot: four valid answers, saved separately.

Both final conditions use the same 120 image IDs and scores: 100 defective and 20 normal images, five per category/folder type. Invalid answers remain in accuracy denominators. Chance is uniform among defect types within the known category, excluding none.

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

## Qualitative examples

Selection rule: first correct-type, first wrong-type and first normal-assigned-none row from the fixed grounded sample. Correct type means the label matches; the wording itself is not independently validated.

- **correct_type** — bottle/contamination; predicted `contamination`; decision `DEFECTIVE`. A central, bright blue and green area with surrounding red and orange patches, indicating a concentrated contamination.
- **wrong_type** — bottle/broken_large; predicted `contamination`; decision `DEFECTIVE`. The image shows a circular area with a ring of colored patches, indicating a contamination on the bottle's surface. The color gradient from blue to orange suggests a localized defect.
- **normal_assigned_none** — bottle/good; predicted `none`; decision `NORMAL`. The image shows a bottle with no visible defects, and the anomaly detector's decision is NORMAL.

## Limitations

- This hardware-limited DINOv2-large at 448 pixels replays the earlier local reproduction; matching that table is not a claim of matching the original paper hardware/model configuration or headline performance.
- Thresholds maximize F1 using the same category test labels: these are oracle thresholds, not independent calibration or deployable operating points. An absent official validation set does not require this choice.
- Only five images per folder type, one fixed sampling seed, and a small 2B model were evaluated. This balanced sample differs from natural benchmark prevalence; no population significance or seed-variance claim is supported.
- Grounded prompts prescribe none for NORMAL. Good-image none rate therefore reflects supplied detector decisions and instructions, not independent recognition. Detector errors propagate.
- The ungrounded condition omits the score, threshold, decision and grounding/normalization instructions. This compares prompt packages, not only the decision token.
- Both conditions see the same resized normalized overlay and known category-level defect taxonomy. Overlay colours have no absolute cross-image anomaly meaning; fine details may be obscured. This is a closed-set experiment.
- Descriptions, locations and severity have no independently reviewed language references. Type correctness is measured against folder names; qualitative text examples do not validate causal explanations or severity.
- The category-subset finding demonstrates global random-stream dependence. Changed exemplars and augmentation streams are confounded; the two runs do not estimate exemplar-only or original-paper uncertainty.

Raw answers and rows are saved in qwen_explanations_grounded.csv / qwen_explanations_ungrounded.csv. Comparison CSV and examples JSON are saved alongside them. No new detector fitting or threshold procedure was introduced beyond the supplied oracle script.

Observed interpretation: grounding changed 2/100 defect-type answers and 1/20 normal-image answers in aggregate. These are descriptive differences, not a demonstrated general improvement. The examples include colour-based rationales; a matching type label does not validate physical or causal explanation quality. The small sample and oracle detector threshold prevent a strong effectiveness claim.

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

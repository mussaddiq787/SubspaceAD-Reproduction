# Qwen explanation experiment protocol — 2026-11-19

The frozen SubspaceAD recipe is replayed across all 15 MVTec AD categories in
its original order, seed 42, DINOv2-registers-large, 448 pixels, k=1, aug_count=30
(with the authors' transistor augmentation exception), PCA explained variance
0.99 and mean layer aggregation. Added score/overlay export does not alter
sampling, features, PCA, scores or metric code. Outputs: `results/qwen_run_full/`.
Compare the complete metric table and selected normal exemplars with the earlier
`results/repro_k1_mvtec_L448/` run; require 1,725 score rows and overlays.

The four-category subset changes globally seeded shuffle/augmentation streams.
Logs demonstrate different screw/transistor exemplars and performance. This is
an observed category-order reproducibility dependence, not a controlled estimate
of exemplar-only effects, seed variance, statistical significance, or uncertainty
in the original publication's numbers. Do not generalize those two differences.

For Qwen, use categories bottle, cable, screw and transistor from the full run.
First run one sampled image per folder type for bottle; preserve that pilot
separately before the final grounded output overwrites the default filename.
The main comparison uses up to five per category/folder type, seeded at 42, with
identical image IDs across grounded and ungrounded conditions. Sample once by
this fixed rule; do not select examples or revise sampling based on answers.

The attached script uses category-wide test labels to choose oracle F1-maximizing
thresholds. Grounded prompts include the score, threshold, NORMAL/DEFECTIVE
result, normalization explanation and an instruction to return none for NORMAL.
Ungrounded prompts omit these detector facts/instructions; therefore this is a
comparison of prompt packages, not an isolated decision-token intervention.
Both use the same normalized overlay and category-level list of defect classes.
That class list is a known evaluation ontology, not open-set recognition.
Individual folder names, labels and revealing paths are excluded from prompts.

Generate greedily at float16 on Turing; record any hardware/API/quantization or
parser modifications. Greedy decoding reduces sampling variability but is not a
guarantee of bit-identical results across environments. Store raw answers. Count
invalid/missing-schema answers honestly; do not silently exclude failures.
Keep the same model, decoding, image preparation and parser for both conditions.

Report detector accuracy on the selected sample; per-category defect-type
accuracy on reference-defective samples (including invalid answers as errors);
uniform chance 1/K for K allowed defect types; good-image none rate; JSON/schema
failure rate; and denominators. A grounded good-image answer is partly prescribed
by its detector decision and prompt. Do not interpret it as independent visual
recognition. Attribute detector errors separately from explanation errors.

One fixed sample and one training seed do not measure stability. Small, balanced
per-type samples differ from the full dataset prevalence. Locations/descriptions/
severity lack expert reference labels and have qualitative examples only; no
validated localization, causal explanation or severity accuracy is claimed.
A two-billion-parameter VLM has limited capacity. Per-image map normalization
removes absolute score information; overlays alone cannot decide defectiveness.

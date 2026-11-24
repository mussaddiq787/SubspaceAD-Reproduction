# Method: score-conditioned Qwen explanations

Each anomaly heatmap is min–max normalized separately for visualization. Its
colours show relative spatial anomaly intensity within that image, not an
absolute defect probability. A normal image can therefore contain a red region;
constant maps normalize to zero, and epsilon prevents an exact maximum of one.

**The overlay alone cannot determine whether an image is defective: Qwen must
always receive the unnormalized image-level `img_score`, and the accept/reject
decision must come from that score against a fixed decision threshold, not from
the heatmap colours.** With the default `mtop1p` aggregation, `img_score` is the
mean of the top 1% of the final unnormalized anomaly map. Other configured score
aggregations must be recorded accurately.

The attached `explain_qwen.py` currently selects an F1-maximizing threshold from
all test labels within each evaluated category. This is an **oracle test-set
threshold**, explicitly disclosed in the script. Consequently, score-conditioned
Qwen results are an exploratory explanation experiment with an optimistic binary
decision supplied to the model, not independent detector validation or evidence
of deployable threshold selection. A future independent evaluation would freeze
thresholds on separate calibration/development data instead.

`per_image_scores.csv` records category, source path, defect folder, reference
label and image score for each test image. Overlays are saved for normal and
defective images when `--save_intro_overlays` is enabled. The run-specific output
directory is nested beneath the requested `--outdir` by the existing runner.

Reference labels and defect folder names are for offline evaluation only. Do not
include them, ground-truth masks, or revealing paths/overlay filenames in Qwen's
input. Pass neutral identifiers, the image/overlay, score and frozen decision
policy instead. This change prepares records and visualizations; it does not
implement or execute Qwen, or introduce a new threshold-selection procedure.

The ungrounded ablation deliberately withholds detector facts and grounding instructions. It is not an authoritative accept/reject mechanism. The measured experiment retained 120 identical samples per condition and reported all failures; see `qwen_evaluation_results.md`.

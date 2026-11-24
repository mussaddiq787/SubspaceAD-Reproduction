"""Arrange existing SubspaceAD images only; no inference or image enhancement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'subspacead-qual-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image

from explain_qwen import best_f1_threshold


def choose(frame, ascending, key='img_score'):
    if frame.empty:
        raise ValueError('No images satisfy a requested selection rule')
    return frame.sort_values([key, 'path'], ascending=[ascending, True], kind='stable').iloc[0].copy()


def load_pixels(path):
    with Image.open(path) as image:
        return np.array(image)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_dir', default='results/qwen_run_full')
    ap.add_argument('--dataset_root', default='datasets/mvtec-ad')
    ap.add_argument('--output', default='report_figs/fig_qual.png')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent
    run_root, dataset = repo / args.run_dir, repo / args.dataset_root
    candidates = sorted(run_root.rglob('per_image_scores.csv'))
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one per-image score CSV, found {len(candidates)}')
    csv_path = candidates[0]
    run = csv_path.parent
    df = pd.read_csv(csv_path)
    if df.path.duplicated().any() or not np.isfinite(df.img_score).all():
        raise ValueError('Duplicate paths or non-finite scores')
    thresholds = {c: best_f1_threshold(g.label, g.img_score) for c, g in df.groupby('category')}
    df['threshold'] = df.category.map(thresholds)
    selected = []
    selected.append((choose(df[(df.category == 'bottle') & (df.defect_type == 'broken_large') & (df.label == 1)], False), 'highest broken_large score'))
    selected.append((choose(df[(df.category == 'screw') & (df.label == 1)], True), 'lowest defective screw score'))
    for category, preferred in [('cable', 'cable_swap'), ('transistor', 'misplaced')]:
        pool = df[(df.category == category) & (df.label == 1)]
        preferred_pool = pool[pool.defect_type == preferred]
        fallback = preferred_pool.empty
        selected.append((choose(pool if fallback else preferred_pool, True),
                         f'lowest {preferred} score' + ('; preferred type absent: used any defect' if fallback else '')))
    normal = df[df.category.isin(['bottle', 'cable', 'screw', 'transistor']) & df.label.eq(0) & df.defect_type.eq('good')].copy()
    if not (normal.threshold > 0).all():
        raise ValueError('Positive category thresholds are required for the score/threshold ranking')
    normal['relative_score'] = normal.img_score / normal.threshold
    selected.append((choose(normal, False, 'relative_score'), 'highest normal score / category threshold'))

    records, panels, fingerprints = [], [], {}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    fingerprints[str(csv_path)] = sha(csv_path)
    for index, (row, rule) in enumerate(selected, 1):
        original_path = repo / Path(row['path'])
        overlay_path = run / 'intro_overlays' / row.category / f'{row.defect_type}_{Path(row["path"]).name}'
        files = [original_path, overlay_path]
        original = load_pixels(original_path)
        overlay = load_pixels(overlay_path)
        if int(row.label) == 0:
            mask_path = None
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            mask_status = 'normal_black_placeholder_no_reference_mask'
        else:
            mask_path = dataset / row.category / 'ground_truth' / row.defect_type / f'{Path(row["path"]).stem}_mask.png'
            if not mask_path.is_file():
                raise FileNotFoundError(f'Missing reference mask for defective image: {mask_path}')
            mask = load_pixels(mask_path)
            files.append(mask_path)
            mask_status = 'existing_reference_mask'
        for path in files:
            fingerprints[str(path)] = sha(path)
        decision = 'DEFECTIVE' if row.img_score >= row.threshold else 'NORMAL'
        outcome = ('true_positive' if decision == 'DEFECTIVE' else 'false_negative') if int(row.label) else ('false_positive' if decision == 'DEFECTIVE' else 'true_negative')
        record = dict(row_number=index, category=row.category, defect_type=row.defect_type,
                      path=row['path'], label=int(row.label), img_score=float(row.img_score),
                      threshold=float(row.threshold), relative_score=float(row.img_score / row.threshold),
                      detector_decision=decision, outcome=outcome, selection_rule=rule,
                      overlay_path=str(overlay_path.relative_to(repo)),
                      ground_truth_path=str(mask_path.relative_to(repo)) if mask_path else '',
                      ground_truth_status=mask_status)
        records.append(record)
        panels.append([original, mask, overlay])

    font = font_manager.findfont('Liberation Serif', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'],
                         'font.size': 6.4, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    width, left, right, col_gap = 3.3, 0.58, 0.04, 0.014
    cell = (width - left - right - 2 * col_gap) / 3
    top, bottom, caption_band, row_gap = 0.23, 0.035, 0.19, 0.012
    height = top + bottom + 5 * (cell + caption_band) + 4 * row_gap
    fig = plt.figure(figsize=(width, height), dpi=300)
    texts = []
    row_labels = []
    for column, title in enumerate(['Input', 'Ground truth', 'SubspaceAD overlay']):
        x = left + column * (cell + col_gap)
        texts.append(fig.text((x + cell / 2) / width, (height - 0.10) / height,
                              title, ha='center', va='center', fontsize=6.4))
    for r, (record, images) in enumerate(zip(records, panels)):
        y = height - top - r * (cell + caption_band + row_gap) - cell
        defect_label = record['defect_type']
        if len(defect_label) > 12 and '_' in defect_label:
            prefix, suffix = defect_label.rsplit('_', 1)
            defect_label = prefix + '_\n' + suffix
        label = fig.text(0.025 / width, (y + cell / 2) / height,
                         f"{record['category']} /\n{defect_label}",
                         ha='left', va='center', fontsize=6.4, linespacing=1.1)
        texts.append(label)
        row_labels.append(label)
        for c, image in enumerate(images):
            x = left + c * (cell + col_gap)
            ax = fig.add_axes([x / width, y / height, cell / width, cell / height])
            if image.ndim == 2:
                ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            else:
                ax.imshow(image, interpolation='antialiased')
            ax.set_box_aspect(1)
            ax.set_axis_off()
        x = left + 2 * (cell + col_gap) + cell / 2
        caption = f"score {record['img_score']:.3g} / thr {record['threshold']:.3g}\n-> {record['detector_decision']}"
        texts.append(fig.text(x / width, (y - 0.022) / height, caption,
                              ha='center', va='top', fontsize=6.0, linespacing=1.0))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in texts:
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < -0.5 or bounds.y0 < -0.5 or bounds.x1 > fig.bbox.width + 0.5 or bounds.y1 > fig.bbox.height + 0.5:
            raise RuntimeError(f'Text would be clipped: {text.get_text()}')
    for label in row_labels:
        if label.get_window_extent(renderer).x1 > (left - 0.01) * fig.dpi:
            raise RuntimeError(f'Row label overlaps image: {label.get_text()}')
    for ax in fig.axes:
        bounds = ax.get_window_extent(renderer)
        if abs(bounds.width - bounds.height) > 0.5:
            raise RuntimeError('Non-square image cell')
    output = repo / args.output
    if output.resolve() in {Path(p).resolve() for p in fingerprints}:
        raise ValueError('Output would overwrite a source image')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor='white')
    plt.close(fig)
    for path, digest in fingerprints.items():
        if sha(path) != digest:
            raise RuntimeError(f'Source content changed: {path}')
    pd.DataFrame(records).to_csv(output.with_suffix('.selection.csv'), index=False)
    metadata = {'score_csv': str(csv_path.relative_to(repo)), 'threshold_method': 'same best_f1_threshold as explain_qwen.py; oracle test-set F1 maximum',
                'normal_ranking': 'img_score / category threshold', 'tie_breaker': 'ascending file path',
                'width_inches': width, 'height_inches': height, 'dpi': 300, 'font': font,
                'cell_size_inches': cell, 'source_files_unchanged': True,
                'source_sha256': fingerprints, 'selected_rows': records,
                'note': 'Purposefully selected score extremes; illustrative examples, not representative performance estimates. Black normal-mask cell is an explicit placeholder.'}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    columns = ['row_number', 'path', 'img_score', 'threshold', 'relative_score', 'detector_decision', 'outcome']
    print(pd.DataFrame(records)[columns].to_string(index=False, float_format=lambda v: f'{v:.8f}'))
    print(f'Figure: {output} ({width:.1f} inches wide, 300 dpi)')
    print(f'Selected-row audit: {output.with_suffix(".selection.csv")}')
    print('PASS: five square-panel rows; existing source files unchanged; no inference or image enhancement.')


if __name__ == '__main__':
    main()

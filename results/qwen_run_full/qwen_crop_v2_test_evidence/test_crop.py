import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
import explain_qwen as q

spec = importlib.util.spec_from_file_location('qwen_v1_reference', str(Path(__file__).with_name('qwen_v1_reference.py')))
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


class CropChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        pixels = np.arange(16 * 12 * 3, dtype=np.uint16).reshape(12, 16, 3) % 180
        self.original = Image.fromarray(pixels.astype(np.uint8))
        self.original_path = self.root / 'original.png'
        self.overlay_path = self.root / 'overlay.png'
        self.original.save(self.original_path)

    def tearDown(self):
        self.temp.cleanup()

    def overlay(self, box=None):
        image = self.original.resize((8, 6))
        if box:
            image.paste((255, 0, 255), box)
        image.save(self.overlay_path)

    def test_margin_mapping_and_original_pixels_only(self):
        self.overlay((2, 1, 4, 3))
        images, data = q.prepare_crop_images(self.original_path, self.overlay_path)
        self.assertEqual(json.loads(data['hotspot_bbox_overlay']), [2, 1, 4, 3])
        self.assertEqual(json.loads(data['crop_bbox_overlay']), [1, 0, 5, 4])
        self.assertEqual(json.loads(data['crop_bbox_original']), [2, 0, 10, 8])
        self.assertEqual(data['hotspot_changed_pixels'], 4)
        self.assertEqual(images[0].tobytes(), self.original.resize((448, 448)).tobytes())
        self.assertEqual(images[1].tobytes(), self.original.crop((2, 0, 10, 8)).resize((448, 448)).tobytes())

    def test_boundary_clipping(self):
        self.overlay((7, 5, 8, 6))
        images, data = q.prepare_crop_images(self.original_path, self.overlay_path)
        self.assertEqual(json.loads(data['crop_bbox_original']), [12, 8, 16, 12])
        self.assertEqual(images[1].size, (448, 448))

    def test_empty_mask_and_336_fallback_size(self):
        self.overlay()
        images, data = q.prepare_crop_images(self.original_path, self.overlay_path, full_size=336)
        self.assertEqual(data['crop_fallback'], 'empty_overlay_difference')
        self.assertEqual(data['hotspot_changed_pixels'], 0)
        self.assertEqual(data['crop_area_fraction'], 1)
        self.assertEqual(json.loads(data['crop_bbox_original']), [0, 0, 16, 12])
        self.assertEqual([im.size for im in images], [(336, 336), (448, 448)])

    def test_noninteger_coordinate_scaling(self):
        self.original.resize((17, 13)).save(self.original_path)
        self.original = Image.open(self.original_path).convert('RGB')
        self.overlay((2, 1, 4, 3))
        _, data = q.prepare_crop_images(self.original_path, self.overlay_path)
        self.assertEqual(json.loads(data['crop_bbox_original']), [2, 0, 11, 9])

    def test_default_prompts_byte_identical(self):
        for grounded in [False, True]:
            for decision in ['NORMAL', 'DEFECTIVE']:
                args = ('bottle', ['broken_large', 'contamination'], decision, 0.12345, 0.23456, grounded)
                self.assertEqual(q.build_prompt(*args), v1.build_prompt(*args))
        crop = q.build_prompt(*args, input_mode='crop')
        self.assertIn('Image 1 is the full product, image 2 is a zoomed view', crop)
        self.assertIn('Describe the physical defect', crop)

    def test_renderer_roundtrip(self):
        from src.subspacead.utils.viz import save_overlay_for_intro
        image = Image.new('RGB', (64, 64), (100, 100, 100))
        image.save(self.original_path)
        anomaly = np.zeros((32, 32), dtype=float)
        anomaly[12:20, 12:20] = 1
        save_overlay_for_intro('test/good/sample.png', image, anomaly, str(self.root), 'bottle')
        overlay = self.root / 'intro_overlays/bottle/good_sample.png'
        _, data = q.prepare_crop_images(self.original_path, overlay)
        self.assertEqual(json.loads(data['hotspot_bbox_overlay']), [10, 10, 22, 22])

    def test_single_image_ask_matches_v1_and_two_image_order(self):
        class Batch(dict):
            def to(self, device): return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return Batch(input_ids=torch.zeros((1, 3), dtype=torch.long))
            def batch_decode(self, values, **kwargs): return [' response ']
        class Model:
            device = 'cpu'
            def generate(self, **kwargs):
                assert kwargs['max_new_tokens'] == 150 and kwargs['do_sample'] is False
                return torch.zeros((1, 4), dtype=torch.long)
        old, new = Processor(), Processor()
        self.assertEqual(v1.ask(Model(), old, self.original, 'prompt'), q.ask(Model(), new, self.original, 'prompt'))
        self.assertEqual(old.messages, new.messages)
        self.assertEqual(old.kwargs, new.kwargs)
        second = Image.new('RGB', (448, 448))
        q.ask(Model(), new, [self.original, second], 'prompt')
        content = new.messages[0]['content']
        self.assertEqual([x['type'] for x in content], ['image', 'image', 'text'])
        self.assertIs(content[0]['image'], self.original)
        self.assertIs(content[1]['image'], second)

    def test_crop_default_reuses_v1_cohort(self):
        class StopBeforeModel(Exception): pass
        output = io.StringIO()
        argv = ['explain_qwen.py', '--run_dir', 'results/qwen_run_full', '--per_type', '5', '--input_mode', 'crop']
        with patch('sys.argv', argv), patch.object(q.torch.cuda, 'is_available', return_value=False), patch.object(q, 'load_model', side_effect=StopBeforeModel), contextlib.redirect_stdout(output):
            with self.assertRaises(StopBeforeModel): q.main()
        self.assertIn('Loaded 493 image scores', output.getvalue())
        self.assertIn('Evaluating 120 images', output.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)

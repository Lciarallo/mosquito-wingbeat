"""Scientific guards for the new objectives, simulations and representations."""
import unittest
import json
import numpy as np
import pandas as pd
import torch
from cross_domain_strategy_models import (StrategyNet, cross_domain_contrast,
    conditional_mmd, mix_known_species, multipitch_features, FEATURE_NAMES)
from train_cross_domain_strategies import calibration_cut, ROOT, OUT
from train_reproduction import sha
from corrected_validation import assert_disjoint
from aedes_binary import targets


class Strategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(6)

    def test_new_representation_is_batch_independent_and_real_duration(self):
        model = StrategyNet(4).eval()
        for size in [2400, 7936]:
            x = torch.randn(2, size)
            with torch.no_grad():
                together = model(x)[0]
                separate = torch.cat([model(a[None])[0] for a in x])
                features = multipitch_features(x)
            torch.testing.assert_close(together, separate, atol=2e-5, rtol=1e-5)
            self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
            self.assertTrue(torch.isfinite(features).all())

    def test_local_spectral_descriptors_preserve_gain_and_dc_invariance(self):
        t = torch.arange(7936) / 8000
        w = (torch.sin(2 * torch.pi * 550 * t) + .4 * torch.sin(2 * torch.pi * 1100 * t))[None]
        torch.testing.assert_close(multipitch_features(w), multipitch_features(.4 * w + .1), atol=.02, rtol=.01)

    def test_contrast_needs_same_species_and_different_domain_and_group(self):
        z = torch.randn(4, 8, requires_grad=True)
        for species, domains, groups in [([0, 1, 2, 3], [0, 1, 0, 1], [0, 1, 2, 3]),
                                         ([0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 2, 3]),
                                         ([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1])]:
            loss, anchors = cross_domain_contrast(z, torch.tensor(species), torch.tensor(domains), torch.tensor(groups))
            self.assertEqual(anchors, 0); self.assertEqual(float(loss.detach()), 0)
        loss, anchors = cross_domain_contrast(z, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), torch.arange(4))
        self.assertEqual(anchors, 4); loss.backward(); self.assertTrue(torch.isfinite(z.grad).all())

    def test_species_conditional_mmd_does_not_merge_different_species(self):
        z = torch.randn(4, 8, requires_grad=True)
        loss = conditional_mmd(z, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]))
        self.assertEqual(float(loss.detach()), 0)

    def test_swarm_donors_have_same_taxon_different_group(self):
        x = torch.randn(6, 2400); species = torch.tensor([0, 0, 1, 1, 2, 3]); group = torch.arange(6)
        out, pairs = mix_known_species(x, species, group, probability=1)
        self.assertEqual(out.shape, x.shape); self.assertEqual(len(pairs), 4)
        for i, j in pairs:
            self.assertEqual(int(species[i]), int(species[j])); self.assertNotEqual(int(group[i]), int(group[j]))
        self.assertTrue(torch.isfinite(out).all())

    def test_cut_cannot_be_fit_on_test_or_drive(self):
        for name in ['test', 'Drive', 'validation']:
            with self.assertRaises(ValueError): calibration_cut(pd.DataFrame(), np.array([]), partition=name)

    def test_frozen_training_uses_only_declared_sources_and_known_labels(self):
        f = pd.read_csv(OUT / 'segments.csv', low_memory=False)
        assert_disjoint(f)
        self.assertEqual(set(f.domain), {'dryad', 'humbug'})
        self.assertEqual(set(f.split), {'train', 'validation', 'calibration'})
        self.assertTrue((targets(f) >= 0).all())
        self.assertFalse(f.path.str.contains('private|download.audio|ufrgs|CDMSC|S_1_D_3', case=False, regex=True).any())

    def test_frozen_checkpoint_selection_was_validation_only(self):
        selection = json.loads((OUT / 'selection.json').read_text())
        candidates = json.loads((OUT / 'validation_search.json').read_text())
        self.assertEqual(selection['selected_model'], max(candidates, key=lambda r: r['selection_score'])['name'])
        for row in candidates:
            self.assertAlmostEqual(row['selection_score'], min(row['validation_cells'].values()) + .05 * np.mean(list(row['validation_cells'].values())))
        frozen = json.loads((OUT / 'freeze.json').read_text())
        for path, h in {**frozen['models_sha256'], **frozen['source_sha256']}.items(): self.assertEqual(sha(ROOT / path), h)

    def test_device_and_original_weights_preserved(self):
        for name in ['device_files_sha256.json', 'original_weights_sha256.json']:
            for path, h in json.loads((OUT / name).read_text()).items(): self.assertEqual(sha(ROOT / path), h)

    def test_constant_pcm_match_is_not_called_duplicate_flight(self):
        from evaluate_cross_domain_strategies import pcmhash
        h, meaningful = pcmhash(np.zeros(7936, dtype='float32'))
        self.assertFalse(meaningful)
        h2, meaningful2 = pcmhash(np.array([0, .01, .1], dtype='float32'))
        self.assertTrue(meaningful2); self.assertNotEqual(h, h2)

    def test_fresh_files_follow_freeze_and_have_no_known_nonconstant_fragments(self):
        frozen = json.loads((OUT / 'freeze.json').read_text())
        plan = json.loads((OUT / 'fresh_selection.json').read_text())
        self.assertGreater(plan['selection_utc'], frozen['frozen_utc'])
        self.assertEqual(plan['models_sha256'], frozen['models_sha256'])
        self.assertEqual(len(plan['files']), 32)
        self.assertEqual(len({item['sha256'] for item in plan['files']}), 32)
        self.assertEqual(sum(item['reference_aedes'] for item in plan['files']), 16)
        for item in plan['files']: self.assertEqual(sha(ROOT / item['path']), item['sha256'])
        audit = pd.read_csv(OUT / 'novelty_audit.csv')
        self.assertTrue((audit.loc[~audit.excluded, 'nonconstant_pcm_matches'] == 0).all())

    def test_all_drive_windows_and_executed_artifacts_are_preserved(self):
        private = pd.read_csv(OUT / 'drive_positive_control.csv')
        self.assertTrue((private.known_windows == 107).all())
        self.assertTrue((private.invalid == 2).all())
        self.assertFalse(private.precision_informative_for_binary_detection.any())
        notebook = json.loads((ROOT / 'estrategias_contrastivas_enxames_2026_10_03.ipynb').read_text())
        code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(code), 3)
        self.assertTrue(all(cell['execution_count'] is not None for cell in code))
        self.assertFalse(any(out['output_type'] == 'error' for cell in code for out in cell['outputs']))


if __name__ == '__main__': unittest.main()

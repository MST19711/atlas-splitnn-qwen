from __future__ import annotations

import unittest

from scripts.qwen35_model_spec import (
    BoundEmbedHeadConfig,
    ModelSpec,
    SplitConfig,
    parse_split,
)


class ParseSplitTests(unittest.TestCase):
    def test_valid_split(self):
        self.assertEqual(parse_split("4,20"), (4, 20))
        self.assertEqual(parse_split("0,24"), (0, 24))
        self.assertEqual(parse_split("1,30"), (1, 30))

    def test_invalid_split_missing_comma(self):
        with self.assertRaises(ValueError):
            parse_split("420")

    def test_invalid_split_too_many_parts(self):
        with self.assertRaises(ValueError):
            parse_split("4,16,4")


class ModelSpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(
            hidden_size=1024,
            vocab_size=248320,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            intermediate_size=3072,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            full_attention_interval=4,
            layer_types=["linear_attention"] * 18 + ["full_attention"] * 6,
            rms_norm_eps=1e-6,
        )

    def test_basic_properties(self):
        self.assertEqual(self.spec.hidden_size, 1024)
        self.assertEqual(self.spec.vocab_size, 248320)
        self.assertEqual(self.spec.num_hidden_layers, 24)

    def test_compute_segment_full_range(self):
        nl_dn, nl_ga = self.spec.compute_segment(0, 24)
        self.assertEqual(nl_dn, 18)
        self.assertEqual(nl_ga, 6)

    def test_compute_segment_dn_only(self):
        nl_dn, nl_ga = self.spec.compute_segment(0, 18)
        self.assertEqual(nl_dn, 18)
        self.assertEqual(nl_ga, 0)

    def test_compute_segment_ga_only(self):
        nl_dn, nl_ga = self.spec.compute_segment(18, 24)
        self.assertEqual(nl_dn, 0)
        self.assertEqual(nl_ga, 6)

    def test_conv_dim_property(self):
        self.assertEqual(self.spec.linear_conv_kernel_dim, 4)

    def test_round_trip_dict(self):
        d = self.spec.to_dict()
        restored = ModelSpec.from_dict(d)
        self.assertEqual(restored.hidden_size, 1024)
        self.assertEqual(restored.vocab_size, 248320)
        self.assertEqual(restored.layer_types, self.spec.layer_types)

    def test_from_dict_with_layer_types(self):
        d = self.spec.to_dict()
        restored = ModelSpec.from_dict(d)
        self.assertEqual(restored.hidden_size, 1024)
        self.assertEqual(restored.num_hidden_layers, 24)
        self.assertEqual(restored.layer_types, self.spec.layer_types)


class SplitConfigTests(unittest.TestCase):
    def test_from_tuple(self):
        cfg = SplitConfig(prefix_end=4, suffix_start=20, total_layers=24)
        self.assertEqual(cfg.prefix_range, (0, 4))
        self.assertEqual(cfg.middle_range, (4, 20))
        self.assertEqual(cfg.suffix_range, (20, 24))

    def test_bound_mode(self):
        cfg = SplitConfig(prefix_end=0, suffix_start=24, total_layers=24)
        self.assertEqual(cfg.prefix_range, (0, 0))
        self.assertEqual(cfg.middle_range, (0, 24))
        self.assertEqual(cfg.suffix_range, (24, 24))

    def test_round_trip_dict(self):
        cfg = SplitConfig(prefix_end=4, suffix_start=20, total_layers=24)
        d = cfg.to_dict()
        restored = SplitConfig.from_dict(d)
        self.assertEqual(restored.prefix_end, 4)
        self.assertEqual(restored.suffix_start, 20)


class BoundEmbedHeadConfigTests(unittest.TestCase):
    def test_create_and_round_trip(self):
        cfg = BoundEmbedHeadConfig(
            tied_weight_path="tied_weight.bin",
            final_norm_path="final_norm.bin",
            dtype="float16",
        )
        self.assertEqual(cfg.tied_weight_path, "tied_weight.bin")
        d = cfg.to_dict()
        restored = BoundEmbedHeadConfig.from_dict(d)
        self.assertEqual(restored.tied_weight_path, "tied_weight.bin")
        self.assertEqual(restored.dtype, "float16")


if __name__ == "__main__":
    unittest.main()

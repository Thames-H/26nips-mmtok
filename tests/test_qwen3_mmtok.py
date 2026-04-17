import importlib
import unittest
from pathlib import Path

import torch


class TestQwen3MMTokScaffolding(unittest.TestCase):
    def test_qwen3_wrapper_export_exists(self):
        module = importlib.import_module("mmtok.qwen")
        self.assertTrue(
            hasattr(module, "mmtok_qwen3_vl"),
            "mmtok.qwen should export mmtok_qwen3_vl",
        )

    def test_extract_question_from_qwen3_messages(self):
        module = importlib.import_module("mmtok.qwen.qwen3_vl_mmtok")
        extract_question_from_messages = getattr(
            module, "extract_question_from_messages"
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy.png"},
                    {"type": "text", "text": "Describe the image."},
                    {"type": "video", "video": "dummy.mp4"},
                    {"type": "text", "text": "Focus on the motion."},
                ],
            },
        ]

        self.assertEqual(
            extract_question_from_messages(messages),
            "Describe the image. Focus on the motion.",
        )

    def test_patch_qwen3_processor_sets_question_from_messages(self):
        module = importlib.import_module("mmtok.qwen.qwen3_vl_mmtok")
        patch_qwen3_vl_processor_for_question_hook = getattr(
            module, "patch_qwen3_vl_processor_for_question_hook"
        )

        class DummyProcessor:
            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=True, **kwargs
            ):
                return "templated"

        class DummyModel:
            def __init__(self):
                self.question = None

            def set_question(self, question):
                self.question = question

        processor = DummyProcessor()
        model = DummyModel()
        patch_qwen3_vl_processor_for_question_hook(processor, model)

        rendered = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "dummy.png"},
                        {"type": "text", "text": "What changed?"},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "templated")
        self.assertEqual(model.question, "What changed?")

    def test_filter_deepstack_by_sequence_indices_keeps_alignment(self):
        module = importlib.import_module("mmtok.qwen.qwen3_VLmodel_mmtok")
        filter_deepstack_by_sequence_indices = getattr(
            module, "_filter_deepstack_by_sequence_indices"
        )

        visual_pos_masks = torch.tensor(
            [[False, True, False, True, True, False]], dtype=torch.bool
        )
        keep_global_indices = torch.tensor([0, 2, 4, 5], dtype=torch.long)
        deepstack_visual_embeds = [
            torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]),
            torch.tensor([[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]]),
        ]

        pruned_mask, pruned_deepstack = filter_deepstack_by_sequence_indices(
            visual_pos_masks=visual_pos_masks,
            keep_global_indices=keep_global_indices,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        self.assertTrue(
            torch.equal(
                pruned_mask,
                torch.tensor([[False, False, True, False]], dtype=torch.bool),
            )
        )
        self.assertEqual(len(pruned_deepstack), 2)
        self.assertEqual(int(pruned_mask.sum().item()), pruned_deepstack[0].shape[0])
        self.assertTrue(
            torch.equal(pruned_deepstack[0], torch.tensor([[30.0, 31.0]]))
        )
        self.assertTrue(
            torch.equal(pruned_deepstack[1], torch.tensor([[60.0, 61.0]]))
        )

    def test_compute_target_vision_tokens_uses_ceil(self):
        module = importlib.import_module("mmtok.qwen.qwen3_VLmodel_mmtok")
        compute_target_vision_tokens = getattr(
            module, "_compute_target_vision_tokens"
        )

        self.assertEqual(compute_target_vision_tokens(10, 1.0), 10)
        self.assertEqual(compute_target_vision_tokens(10, 0.21), 3)
        self.assertEqual(compute_target_vision_tokens(10, 0.01), 1)
        self.assertEqual(compute_target_vision_tokens(10, 0.0), 0)

    def test_build_keep_indices_preserves_sorted_text_image_video_order(self):
        module = importlib.import_module("mmtok.qwen.qwen3_VLmodel_mmtok")
        build_keep_indices = getattr(module, "_build_keep_indices")

        class DummyConfig:
            image_token_id = 11
            video_token_id = 22

        input_ids = torch.tensor([101, 11, 11, 202, 22, 22, 303], dtype=torch.long)
        keep_indices = build_keep_indices(
            input_ids=input_ids,
            image_keep_local=torch.tensor([1], dtype=torch.long),
            video_keep_local=torch.tensor([0], dtype=torch.long),
            config=DummyConfig(),
        )

        self.assertTrue(
            torch.equal(keep_indices, torch.tensor([0, 2, 3, 4, 6], dtype=torch.long))
        )

    def test_qwen3_examples_exist(self):
        repo_root = Path(__file__).resolve().parents[1]
        expected = [
            repo_root / "example" / "qwen3_mmtok_image_example.py",
            repo_root / "example" / "qwen3_mmtok_video_example.py",
            repo_root / "example" / "lmms_eval_qwen3_mmtok.py",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing example file: {path}")


if __name__ == "__main__":
    unittest.main()

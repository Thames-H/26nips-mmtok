import re
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from mmtok.qwen import mmtok_qwen3_vl

process_vision_info, _has_qwen_vl = optional_import(
    "qwen_vl_utils", "process_vision_info"
)
if not _has_qwen_vl:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`"
    )


@register_model("qwen3_vl_mmtok")
class Qwen3_VL_MMTok(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        retain_ratio: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        assert int(batch_size) == 1, "Qwen3-VL MMTok currently supports batch_size=1 only."
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        match = re.search(r"A\d+B", pretrained)
        model_cls = (
            Qwen3VLMoeForConditionalGeneration
            if match
            else Qwen3VLForConditionalGeneration
        )
        self._model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        self._model, self.processor = mmtok_qwen3_vl(
            self._model,
            language_tokenizer=self._tokenizer,
            processor=self.processor,
            retain_ratio=retain_ratio,
        )

        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.reasoning_prompt = (
            reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.use_cache = use_cache
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)

        if accelerator.num_processes > 1:
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3-VL.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get(
                "until", [self.tokenizer.decode(self.eot_token_id)]
            )
            if isinstance(until, str):
                until = [until]

            batched_messages = []
            for idx, context in enumerate(contexts):
                context = context.replace("<image>", "")
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt

                content = []
                for visual in visual_list[idx]:
                    if isinstance(visual, str) and visual.endswith(
                        (".mp4", ".avi", ".mov")
                    ):
                        content.append(
                            {
                                "type": "video",
                                "video": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        content.append(
                            {
                                "type": "image",
                                "image": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                content.append({"type": "text", "text": context})
                batched_messages.append(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ]
                )

            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=False,
            )
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(
                    0, total_frames - 1, self.max_num_frames, dtype=int
                )
                indices = np.unique(indices)
                video_inputs[0] = video_inputs[0][indices]

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs_qwen,
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.device if self.device_map != "auto" else "cuda")

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), clean_ans
                )
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

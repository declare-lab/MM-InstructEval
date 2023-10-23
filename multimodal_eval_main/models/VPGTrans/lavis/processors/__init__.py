"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from multimodal_eval_main.models.VPGTrans.lavis.processors.base_processor import BaseProcessor

from multimodal_eval_main.models.VPGTrans.lavis.processors.alpro_processors import (
    AlproVideoTrainProcessor,
    AlproVideoEvalProcessor,
)
from multimodal_eval_main.models.VPGTrans.lavis.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from multimodal_eval_main.models.VPGTrans.lavis.processors.gpt_processors import (
    GPTVideoFeatureProcessor,
    GPTDialogueProcessor,
)
from multimodal_eval_main.models.VPGTrans.lavis.processors.clip_processors import ClipImageTrainProcessor

from multimodal_eval_main.models.VPGTrans.lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    # ALPRO
    "AlproVideoTrainProcessor",
    "AlproVideoEvalProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    "ClipImageTrainProcessor",
    # GPT
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor

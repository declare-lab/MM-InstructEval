import json
import pickle
import random
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
import torch
from PIL import Image
from fire import Fire
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.transforms import Compose
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)



from multimodal_eval_main.data_loading import MultimodalSequence, MultimodalPart
from multimodal_eval_main.models.fromage.models import load_fromage, Fromage
from multimodal_eval_main.models.lavis_models.blip2_t5_instruct import Blip2T5Instruct
from multimodal_eval_main.models.lavis_models.blip_processors import BlipImageEvalProcessor
from multimodal_eval_main.models.Multimodal_GPT.app import Inferencer
from multimodal_eval_main.models.open_flamingo import create_model_and_transforms, Flamingo



import numpy as np

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_image(path_or_url: str) -> Image:
    is_url = urlparse(path_or_url).scheme != ""
    if is_url:
        response = requests.get(path_or_url)
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(path_or_url)


def make_fn(labels: List[str], tokenizer: PreTrainedTokenizer):
    token_ids = [tokenizer(x, add_special_tokens=False).input_ids[0] for x in labels]

    def fn(*args, **kwargs):
        assert args is not None
        assert kwargs is not None
        return token_ids

    return fn


class MultimodalModel(BaseModel, arbitrary_types_allowed=True):
    max_input_length: int = 2048
    max_output_length: int = 100
    do_sample: bool = False

    def run(self, context: MultimodalSequence, labels: List[str] = None) -> str:
        raise NotImplementedError


class FromageModel(MultimodalModel):
    model_dir: str = "fromage_model"
    model: Optional[Fromage]

    @staticmethod
    def download_checkpoint(
        url: str = "https://github.com/chiayewken/multimodal-inference/releases/download/v0.1.0/fromage_model.zip",
        path: str = ".",
        folder: str = "fromage_model",
        embed_path: str = "cc3m_embeddings.pkl",
    ):
        download_and_extract_archive(url, download_root=path)
        assert Path(folder).exists()

        # Write dummy image embeddings needed to load model
        data = dict(paths=[""], embeddings=torch.zeros(1, 256))
        with open(Path(folder, embed_path), "wb") as f:
            pickle.dump(data, f)

    def load(self):
        if self.model is None:
            self.download_checkpoint(
                path=str(Path(self.model_dir).parent.resolve()),
                folder=self.model_dir,
            )
            self.model = load_fromage(self.model_dir)
            

    def run(self, context: MultimodalSequence, labels: List[str] = None) -> str:
        inputs = []
        for part in context.parts:
            if part.is_image:
                image = get_image(part.content)
                image = image.resize((224, 224))
                image = image.convert("RGB")
                inputs.append(image)
            else:
                inputs.append(part.content)

        self.load()
        parameters = get_parameter_number(self.model)
        # print("+++++++++++++++++++++++++++++++++++++++++++=") 
        # print(parameters)
        try:
            outputs = self.model.generate_for_images_and_texts(
                inputs,
                num_words=self.max_output_length,
                ret_scale_factor=0.0,  # Don't generate images
                max_input_length=self.max_input_length,
                labels=labels,
                temperature=1.0 if self.do_sample else 0.0,
            )
        except Exception as e:
            outputs = [str(e)]

        return outputs[0]


def test_fromage(
    image_paths: List[str] = (
        "https://i.pinimg.com/736x/d3/8c/21/d38c21ca670ce0be2d01c301b1f0e7d3--vintage-dior-vintage-dresses.jpg",
        "https://secure.img1-cg.wfcdn.com/im/06305386/compr-r85/1695/169530576/ainsley-736-vegan-leather-sofa.jpg",
    ),
    prompts: List[str] = (
        "Q: What is this image?\nA:",
        "Q: What color is the dress?\nA:",
        "Q: When do you think it was taken?\nA:",
        "Q: What color is the sofa?\nA:",
        "Q: What is the difference between the two images?\nA:",
    ),
):
    print(json.dumps(locals(), indent=2))
    parts = [MultimodalPart(content=path, is_image=True) for path in image_paths]
    model = FromageModel()

    for p in prompts:
        outputs = model.run(
            context=MultimodalSequence(parts=parts + [MultimodalPart(content=p)]),
        )
        print(dict(prompt=p, outputs=outputs))


class OpenFlamingoModel(MultimodalModel):
    model_type = 'OpenFlamingo-9B'
    device: str = "cuda"
    path_encoder: str = "huggyllama/llama-7b"
    path_model: str = "/data/xiaocui/weights/openflamingo/OpenFlamingo-9B/checkpoint.pt"
    model: Optional[Flamingo]
    processor: Optional[Compose]
    tokenizer: Optional[PreTrainedTokenizer]

    def load(self):
        if self.model is None:
            model, processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path=self.path_encoder,
                tokenizer_path=self.path_encoder,
                cross_attn_every_n_layers=4,
            )
            repo = str(Path(self.path_model).parent)
            filename = Path(self.path_model).name
            # checkpoint_path = hf_hub_download(repo, filename)
            model.load_state_dict(torch.load(self.path_model), strict=False)
            tokenizer.padding_side = "left"  # For generation
            self.model, self.tokenizer, self.processor = model, tokenizer, processor
            self.model.to(self.device)
            parameters = get_parameter_number(self.model)
            print("+++++++++++++++++++++++++++++++++++++++++++=") 
            print(parameters)

    def run(self, context: MultimodalSequence, labels: List[str] = None) -> str:
        self.load()
        images = [get_image(part.content) for part in context.parts if part.is_image]
        x_image = torch.stack([self.processor(i) for i in images])
        x_image = x_image.unsqueeze(1).unsqueeze(0)

        text = ""
        for part in context.parts:
            if part.is_image:
                text = text + "<image>"
            elif text == "":
                text = part.content
            else:
                text = text + part.content

        x_text = self.tokenizer(
            [text],
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )

        tokens_fn = None if labels is None else make_fn(labels, self.tokenizer)
        ##task != "QA":
        ##task == "QA", self.max_output_length=50
        outputs = self.model.generate(
            vision_x=x_image.to(self.device),
            lang_x=x_text["input_ids"].to(self.device),
            attention_mask=x_text["attention_mask"].to(self.device),
            max_new_tokens=self.max_output_length,
            prefix_allowed_tokens_fn=tokens_fn,
            do_sample=self.do_sample,
        )

        _, length = x_text.input_ids.shape
        outputs = outputs[:, slice(length, len(outputs[0]))]
        return self.tokenizer.decode(outputs[0])


# noinspection HttpUrlsUsage
def test_flamingo():
    model = OpenFlamingoModel()
    parts = [
        MultimodalPart(
            content="http://images.cocodataset.org/val2017/000000039769.jpg",
            is_image=True,
        ),
        MultimodalPart(content="An image of two cats.<|endofchunk|>"),
        MultimodalPart(
            content="http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            is_image=True,
        ),
        MultimodalPart(content="An image of a bathroom sink.<|endofchunk|>"),
        MultimodalPart(
            content="http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
            is_image=True,
        ),
        MultimodalPart(content="An image of"),
    ]
    print(dict(input=parts))
    print(model.run(context=MultimodalSequence(parts=parts)))


class SeqToSeqModel(MultimodalModel):
    path_model: str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    device: str = "cuda"
    load_8bit: bool = False

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path_model, **args)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path_model)

    def run(self, context: MultimodalSequence, labels: List[str] = None) -> str:
        self.load()
        prompt = "".join([p.content for p in context.parts if not p.is_image])
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        tokens_fn = None if labels is None else make_fn(labels, self.tokenizer)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            prefix_allowed_tokens_fn=tokens_fn,
            do_sample=self.do_sample,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)




class BlipModel(MultimodalModel):
    path_model: str = "Salesforce/blip2-flan-t5-xxl"
    # path_model: str = "/home/xiaocui/.cache/huggingface/hub/ models--Salesforce--blip2-flan-t5-xxl"
    processor: Optional[Blip2Processor]
    model: Optional[Blip2ForConditionalGeneration]
    model: Optional[PreTrainedModel]
    device: str = "cuda"
    load_8bit: bool = True

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            path = self.path_model
            self.processor = Blip2Processor.from_pretrained(path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(path, **args)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)

    def run(self, context: MultimodalSequence, labels: List[str] = None) -> str:
        prompt = "".join([p.content for p in context.parts if not p.is_image])
        image = None
        for p in context.parts:
            if p.is_image:
                image = get_image(p.content).convert("RGB")
                break

        self.load()
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        # noinspection PyTypeChecker
        inputs = inputs.to(torch.float16)  # Required by model internally
        # noinspection PyUnresolvedReferences
        tokenizer = self.processor.tokenizer
        tokens_fn = None if labels is None else make_fn(labels, tokenizer)

        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            prefix_allowed_tokens_fn=tokens_fn,
            do_sample=self.do_sample,
        )
        return self.processor.decode(outputs[0], skip_special_tokens=True)


def test_blip():
    model = BlipModel()
    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    question = "how many dogs are in the picture?"
    context = MultimodalSequence(
        parts=[
            MultimodalPart(content=url, is_image=True),
            MultimodalPart(content=question),
        ]
    )
    print(model.run(context))


class RandomModel(MultimodalModel):
    random_seed: int = 42
    is_seeded: bool = False

    def load(self):
        if not self.is_seeded:
            random.seed(self.random_seed)
            self.is_seeded = True

    def run(self, context: MultimodalSequence, labels: List[str] = None) -> str:
        self.load()
        return random.choice(labels)


def test_download():
    snapshot_download(
        repo_id="TheBloke/vicuna-13B-1.1-HF",
        local_dir="llm/vicuna-13b",
        local_dir_use_symlinks=True,
    )


class InstructBlipT5Model(MultimodalModel):
    path_model: str = "flant5xxl"
    processor: Optional[BlipImageEvalProcessor]
    model: Optional[Blip2T5Instruct]
    device: str = "cuda"
    image_size: int = 224
    

    def load(self):
        # print(f"-----------------self.path_model is {self.path_model}----------------------")
        if self.model is None:
            self.model = Blip2T5Instruct.from_pretrained(self.path_model)
            
            self.model.eval()
            self.model.to(self.device)
            parameters = get_parameter_number(self.model)
            print("+++++++++++++++++++++++++++++++++++++++++++=") 
            print(parameters)
        if self.processor is None:
            self.processor = BlipImageEvalProcessor(image_size=self.image_size)

    def run(self, context: MultimodalSequence, labels: List[str] = None) -> str:
        image = None
        for p in context.parts:
            if p.is_image:
                image = get_image(p.content).convert("RGB")
                break

        self.load()
        inputs = dict(
            image=self.processor(image).unsqueeze(0).to(self.device),
            prompt="".join([p.content for p in context.parts if not p.is_image and p.content]),
        )
        tokenizer = self.model.t5_output_tokenizer
        tokens_fn = None if labels is None else make_fn(labels, tokenizer)
        outputs = self.model.generate(
            inputs,
            temperature=1.0 if self.do_sample else 0.0,
            num_beams=1,
            top_p=1.0,
            max_length=self.max_output_length,
            tokens_fn=tokens_fn,
        )
        return outputs[0]
    

def test_instruct_blip():
    model = InstructBlipT5Model()
    url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    inputs = MultimodalSequence(
        parts=[
            MultimodalPart(content=url, is_image=True),
            MultimodalPart(content="What is unusual about this image?"),
        ]
    )
    outputs = model.run(inputs)
    print(dict(outputs=outputs))
    breakpoint()



    

def select_model(model_name: str, **kwargs):
    if model_name == "flan_t5":
        return SeqToSeqModel(path_model="google/flan-t5-xxl", load_8bit=True, **kwargs)
    if model_name == "flamingo":
        return OpenFlamingoModel(**kwargs)
    if model_name == "fromage":
        return FromageModel(**kwargs)
    if model_name == "seq_to_seq":
        return SeqToSeqModel(**kwargs)
    if model_name == "blip":
        return BlipModel(**kwargs)
    if model_name == "instruct_blip_t5":
        return InstructBlipT5Model(**kwargs)
    if model_name == "random":
        return RandomModel(**kwargs)
    raise KeyError(model_name)


if __name__ == "__main__":
    test_instruct_blip()

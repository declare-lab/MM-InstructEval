import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional

from fire import Fire
from pydantic import BaseModel, Field


class MultimodalPart(BaseModel):
    is_image: bool = False
    content: Optional[str] = None
    few_shot_samples: Optional[list] = None
    


class MultimodalSequence(BaseModel):
    parts: List[MultimodalPart]


def fix_image_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.with_suffix(".jpg").exists():
        print(dict(orig=path, fixed=".jpg"))
        return path.with_suffix(".jpg")
    if path.with_suffix(".png").exists():
        print(dict(orig=path, fixed=".png"))
        return path.with_suffix(".png")
    raise ValueError(path)


def parse_exam_text(text: str, image_dir: str) -> MultimodalSequence:
    marker = "(image)"
    parts = []
    frontier = -1
    temp = []

    for start in range(len(text)):
        if text[start:].startswith(marker):
            if temp:
                parts.append("".join(temp))
                temp = []
            end = text[start:].index("]") + start
            parts.append(text[slice(start, end + 1)])
            frontier = end
        elif start > frontier:
            temp.append(text[start])

    if temp:
        parts.append("".join(temp))
    assert "".join(parts) == text

    sequence = MultimodalSequence(parts=[])
    for p in parts:
        if p.startswith(marker):
            content = p[slice(len(marker), len(p))]
            content = content.strip(" ()[]")
            path = Path(image_dir, content)
            path = fix_image_path(path)
            sequence.parts.append(MultimodalPart(content=str(path), is_image=True))
        else:
            sequence.parts.append(MultimodalPart(content=p, is_image=False))
    return sequence


def test_parse_exam(image_dir: str = "images"):
    texts = [
        "(image)[image-296.png]\n\nIn the United States",
        "found in most cells.\n\n(image)[image-434.png]\n\nThe section",
        "found in most cells.\n\n(image)[image-434.png]\n\nThe section(image)[image-434.png]",
        "(1) (image)[image-542.jpg]",
        "Which graph best represents the relationship between photon energy and photon wavelength?",
    ]
    for t in texts:
        print(repr(t))
        print(parse_exam_text(t, image_dir=image_dir).json(indent=2))


class ExamSample(BaseModel):
    question_text: str
    background_description: List[str]
    raw_options: List[str] = Field(alias="options")
    answer_text: str
    need_image: str
    language: str
    level: str
    subject: str
    subject_category: str
    year: str
    prompt: str = ""
    num_images: int = -1
    pred: str = ""
    gold: str = ""
    model_name: str = ""

    @property
    def options(self) -> List[str]:
        outputs = []
        for o in self.raw_options:
            if o == "nan":
                continue

            prefix, text = o.split(maxsplit=1)
            assert prefix.startswith("(") and prefix.endswith(")")
            outputs.append(text)

        return outputs

    def get_possible_labels(self) -> List[str]:
        labels = list("ABCD")
        labels = labels[: len(self.options)]
        return labels

    def as_prompt(self, image_dir: str, include_answer=True) -> MultimodalSequence:
        parts = []
        topic = self.subject_category
        hint = MultimodalPart(
            content=f"The following are multiple-choice questions about {topic}\n",
            is_image=False,
        )
        parts.append(hint)

        for text in self.background_description:
            if text.strip() != "":
                parts.extend(parse_exam_text(text, image_dir).parts)
            parts.append(MultimodalPart(content="\n"))

        parts.extend(parse_exam_text(self.question_text, image_dir).parts)
        parts.append(MultimodalPart(content="\n"))

        labels = self.get_possible_labels()
        for i, o in enumerate(self.options):
            parts.append(MultimodalPart(content=f"{labels[i]}. "))
            parts.extend(parse_exam_text(o, image_dir).parts)
            parts.append(MultimodalPart(content="\n"))

        parts.append(MultimodalPart(content="Answer: "))

        if include_answer:
            parts.append(MultimodalPart(content=f"{self.get_answer_label()}\n\n"))

        return MultimodalSequence(parts=parts)

    def get_answer_label(self) -> str:
        i = 0
        for j, o in enumerate(self.raw_options):
            prefix = o.split()[0]
            if self.answer_text in prefix:
                i = j
                break

        labels = self.get_possible_labels()
        return labels[i]

    @property
    def image_type(self):
        marker = "(image)"
        parts = []

        if marker in " ".join(self.options):
            parts.append("options")
        if marker in self.question_text:
            parts.append("question")
        if marker in " ".join(self.background_description):
            parts.append("background")
        return "-".join(parts)

    def count_images(self) -> int:
        parts = [
            *self.raw_options,
            self.question_text,
        ]
        parts.extend(self.background_description)
        return " ".join(parts).count("(image)")


class ExamData(BaseModel):
    samples: List[ExamSample]

    @classmethod
    def load(cls, path: str):
        samples = []
        with open(path) as f:
            for raw in json.load(f):
                samples.append(ExamSample(**raw))
        return cls(samples=samples)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.samples:
                print(s.json(), file=f)

    def analyze(self, image_dir: str):
        random.seed(0)
        for s in random.sample(self.samples, k=3):
            print(s.json(indent=2))
            print(s.as_prompt(image_dir=image_dir, include_answer=True).json(indent=2))

        for s in self.samples:
            assert s.as_prompt(image_dir, include_answer=True) is not None

        info = dict(
            samples=len(self.samples),
            need_image=Counter(s.need_image for s in self.samples),
            image_type=Counter(s.image_type for s in self.samples),
            num_images=Counter(s.count_images() for s in self.samples),
            num_options=Counter(len(s.options) for s in self.samples),
            answers=Counter(s.get_answer_label() for s in self.samples),
        )
        print(json.dumps(info, indent=2))


def test_data(path: str = "english-questions-image.json", image_dir="images-english"):
    data = ExamData.load(path)
    data.analyze(image_dir)
    breakpoint()


if __name__ == "__main__":
    Fire()

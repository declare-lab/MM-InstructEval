## Multimodal Content Comprehension Tasks


We conduct comprehensive evaluation of various Language Models across a diverse range of multimodal content comprehension tasks, including **MSA, MABSA, MHMR, MSR, MRE, and VQA**. Detailed statistics for each task and the datasets can be found in the following Table.


|Task| Dataset |Data Link | Paper | Modality | Test | Labels |
|----|---------|----------|-------|----------|-------|-------|
|MSA |MVSA-Single |[Data](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)| [Paper](https://link.springer.com/chapter/10.1007/978-3-319-27674-8_2)    |Text-Image | 413 | 3 |
|MSA |MVSA-Multiple|[Data](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)| [Paper](https://link.springer.com/chapter/10.1007/978-3-319-27674-8_2)    |Text-Image | 413 | 3 |
|MSA| TumEmo | [Data](https://github.com/YangXiaocui1215/MVAN) | [Paper](https://doi.org/10.1109/TMM.2020.3035277) |Text-Image | 9463  | 7 |
|MSA| MOSI-2 |[Data](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) | [Paper](https://arxiv.org/abs/1606.06259) | Video | 654 | 2 |
|MSA| MOSI-7 |[Data](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) | [Paper](https://arxiv.org/abs/1606.06259) | Video | 684 | 7 |
|MSA| MOSEI-2 |[Data](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) | [Paper](https://aclanthology.org/P18-1208/) | Video | 2797 | 2 |
|MSA | MOSEI-7 |[Data](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) | [Paper](https://aclanthology.org/P18-1208/) | Video | 3588 | 7 |
|MABSA | Twitter-2015 | [Data](https://github.com/jefferyYu/TomBERT) | [Paper](http://www.qizhang.info/paper/aaai2017-twitterner.pdf) |Text-Image |1037 | 3 |
|MABSA | Twitter-2017 | [Data](https://github.com/jefferyYu/TomBERT) | [Paper](https://aclanthology.org/P18-1185/)|Text-Image |1234 | 3 |
|MABSA | MASAD | [Data](https://github.com/DrJZhou/MASAD) | [Paper](https://www.sciencedirect.com/science/article/pii/S0925231221007931?via%3Dihub)|Text-Image |4935 | 2 |
|MHMR | Hate | [Data](https://github.com/facebookresearch/fine_grained_hateful_memes) | [Paper](https://aclanthology.org/2021.woah-1.21/) |Text-Image | 500 | 2 |
|MSR | Sarcasm | [Data](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection) | [Paper](https://aclanthology.org/P19-1239/)|Text-Image | 2409 | 2 |
|MRE | MNRE | [Data](https://github.com/thecharm/Mega) | [Paper](https://dl.acm.org/doi/10.1145/3474085.3476968) |Text-Image | 640 | 19 |
|VQA | ScienceQA | [Data](https://scienceqa.github.io/#dataset) | [Paper](https://proceedings.neurips.cc//paper_files/paper/2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html) |Text-Image | 2017 | - |


## Download Multimodal Datasets

You can download the processed multimodal datasets by the [Google Drive](https://drive.google.com/drive/folders/1VbnmfzSV_igvf-R70oJx_ppE4O0KfK1p?usp=sharing).


## Data Files Structure
```
├─multimodal_data
│ ├─task_file
│     └─dataset_file
│           |-test.csv
│           |-test.json
│           └─image_data 
│               └─test_image
│                   |-1.jpg
│                   |-2.jpg
│                   ...
                   
For Example,
├─multimodal_data
│ ├─MSA
│    |-TumEmo
│    |-MVSA-Multiple
│    └─MVSA-Single
│           |-test.csv
│           |-test.json
│           └─image_data 
│               └─test_image
│                   |-1.jpg
│                   |-2.jpg
│                   ...
│ ├─MABSA
│ ├─QA

...

```

---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: img_type
    dtype: string
  - name: format_type
    dtype: string
  - name: task
    dtype: string
  - name: source
    dtype: string
  - name: image
    sequence: image
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test
    num_bytes: 1887306946.625
    num_examples: 7211
  download_size: 1840289781
  dataset_size: 1887306946.625
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---


<p align="left">
  <a href="https://github.com/fudan-zvg/spar.git">
    <img alt="GitHub Code" src="https://img.shields.io/badge/Code-spar-black?&logo=github&logoColor=white" />
  </a>
  <a href="https://arxiv.org/abs/2503.22976">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-spar-red?logo=arxiv" />
  </a>
  <a href="https://fudan-zvg.github.io/spar">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-spar-blue" />
  </a>
</p>

# 🎯 Spatial Perception And Reasoning Benchmark (SPAR-Bench)
> A benchmark to evaluate **spatial perception and reasoning** in vision-language models (VLMs), with high-quality QA across 20 diverse tasks.

**SPAR-Bench** is a high-quality benchmark for evaluating spatial perception and reasoning in vision-language models (VLMs). It covers 20 diverse spatial tasks across single-view, multi-view, and video settings, with a total of **7,207 manually verified QA pairs**.

SPAR-Bench is derived from the large-scale [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M) dataset, and specifically designed to support **zero-shot evaluation** and **task-specific analysis**


> 📌 SPAR-Bench at a glance:
> - ✅ 7,207 manually verified QA pairs
> - 🧠 20 spatial tasks (depth, distance, relation, imagination, etc.)
> - 🎥 Supports single-view, multi-view inputs
> - 📏 Two evaluation metrics: Accuracy & MRA
> - 📷 Available in RGB-only and RGB-D versions


## 🧱 Available Variants

**We provide four versions of SPAR-Bench**, covering both RGB-only and RGB-D settings, as well as full-size and lightweight variants:

| Dataset Name                             | Description                                                        |
|------------------------------------------|--------------------------------------------------------------------|
| [`SPAR-Bench`](https://huggingface.co/datasets/jasonzhango/SPAR-Bench)                 | Full benchmark (7,207 QA) with RGB images                         |
| [`SPAR-Bench-RGBD`](https://huggingface.co/datasets/jasonzhango/SPAR-Bench-RGBD)       | Full benchmark with depths, camera pose and intrinsics              |
| [`SPAR-Bench-Tiny`](https://huggingface.co/datasets/jasonzhango/SPAR-Bench-Tiny)       | 1,000-sample subset (50 QA per task), for fast evaluation or APIs |
| [`SPAR-Bench-Tiny-RGBD`](https://huggingface.co/datasets/jasonzhango/SPAR-Bench-Tiny-RGBD) | Tiny version with RGBD inputs                                     |

> 🔎 Tiny versions are designed for quick evaluation (e.g., APIs, human studies).  
> 💡 RGBD versions include depths, poses, and intrinsics, suitable for 3D-aware models.

To load a different version via `datasets`, simply change the dataset name:

```python
from datasets import load_dataset
spar = load_dataset("jasonzhango/SPAR-Bench")
spar_rgbd = load_dataset("jasonzhango/SPAR-Bench-RGBD")
spar_tiny = load_dataset("jasonzhango/SPAR-Bench-Tiny")
spar_tiny_rgbd = load_dataset("jasonzhango/SPAR-Bench-Tiny-RGBD")
```
## 🕹️ Evaluation

SPAR-Bench supports two evaluation metrics, depending on the question type:

- **Accuracy** – for multiple-choice questions (exact match)
- **Mean Relative Accuracy (MRA)** – for numerical-answer questions (e.g., depth, distance)


> 🧠 The MRA metric is inspired by the design in [Thinking in Space](https://github.com/vision-x-nyu/thinking-in-space), and is tailored for spatial reasoning tasks involving quantities like distance and depth.

We provide an **evaluation pipeline** in our [GitHub repository](https://github.com/hutchinsonian/spar), built on top of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). 


## 📚 Bibtex

If you find this project or dataset helpful, please consider citing our paper:

```bibtex
@article{zhang2025from,
    title={From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D},
    author={Zhang, Jiahui and Chen, Yurui and Zhou, Yanpeng and Xu, Yueming and Huang, Ze and Mei, Jilin and Chen, Junhui and Yuan, Yujie and Cai, Xinyue and Huang, Guowei and Quan, Xingyue and Xu, Hang and Zhang, Li},
    year={2025},
    journal={arXiv preprint arXiv:2503.22976},
}
```

<!-- ## 📄 License

This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

You may use, share, modify, and redistribute this dataset **for any purpose**, including commercial use, as long as proper attribution is given.

[Learn more](https://creativecommons.org/licenses/by/4.0/) -->

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

# 🎯 SPAR-Bench-Tiny

> A lightweight subset of SPAR-Bench for **fast evaluation** of spatial reasoning in vision-language models (VLMs).

**SPAR-Bench-Tiny** contains **1,000 manually verified QA pairs** — 50 samples per task across **20 spatial tasks** — covering single-view and multi-view inputs. 

This dataset mirrors the structure and annotation of the full [SPAR-Bench](https://huggingface.co/datasets/jasonzhango/SPAR-Bench), but is **10× smaller**, making it ideal for low-latency evaluation.



## 📥 Load with `datasets`

```python
from datasets import load_dataset
spar_tiny = load_dataset("jasonzhango/SPAR-Bench-Tiny")
```
## 🕹️ Evaluation

SPAR-Bench-Tiny uses the **same evaluation protocol and metrics** as the full [SPAR-Bench](https://huggingface.co/datasets/jasonzhango/SPAR-Bench).

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
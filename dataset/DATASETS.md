# VLM 评测数据集说明

本文档整理自 `dataset/加载&下载数据集.ipynb` 中的介绍，并补充了各数据集自带
README 中的官方链接。三个脚本只负责**下载**（CV-Bench 另外负责**样例提取**），
不包含模型推理或评测流程。

| 数据集 | Hugging Face 仓库 | 对应脚本 | 本地目录 |
| --- | --- | --- | --- |
| CV-Bench | `nyu-visionx/CV-Bench` | `download_cv_bench.py` | `dataset/CV-Bench/` |
| SPAR-Bench | `jasonzhango/SPAR-Bench` | `download_spar_bench.py` | `dataset/SPAR-Bench/` |
| VSI-Bench | `nyu-visionx/VSI-Bench` | `download_vsi_bench.py` | `dataset/VSI-Bench/` |

---

## 1. CV-Bench

### 数据集简介

CV-Bench 来自 Cambrian-1 研究，核心是测试 VLM 的基础视觉能力。

例如，给模型一张室内图片，提问：

> 图中的椅子和桌子，哪个距离摄像机更近？

**优点**：有图像、问题、选项、正确答案，可直接计算 Accuracy；还可以分别加载
2D 和 3D 子集，不必进行额外标注。

**局限**：主要是从 RGB 图像推断深度关系，而不是直接读取深度图；不考察多帧记忆。

### 主要任务

包含四种典型任务：

- **2D 空间关系**：左、右、上、下等；
- **物体数量识别**；
- **3D 深度顺序**：哪个物体更靠近相机；
- **相对距离**：比较物体与相机、或物体之间的距离关系。

其中 2D 与 3D 对应数据集的两个子集配置（`2D`、`3D`），脚本分别处理，不做合并或删减。

### 官方地址

- Hugging Face 数据集：<https://huggingface.co/datasets/nyu-visionx/CV-Bench>
- 论文（Cambrian-1）：<https://arxiv.org/abs/2406.16860>
- 项目主页：<https://cambrian-mllm.github.io/>
- 代码仓库：<https://github.com/cambrian-mllm/cambrian>

### 当前项目中的数据位置

```text
dataset/CV-Bench/
├── test_2d.parquet          # 2D 子集（图片内嵌在 parquet 中）
├── test_3d.parquet          # 3D 子集
├── README.md
├── build_img.py             # 数据集仓库自带的辅助脚本
└── samples/                 # 脚本生成的样例
    ├── 2D/
    │   ├── 2d_01.png ... 2d_30.png
    │   └── annotations.csv
    ├── 3D/
    │   ├── 3d_01.png ... 3d_30.png
    │   └── annotations.csv
    ├── 2d_overview.jpg
    └── 3d_overview.jpg
```

### 对应处理脚本

`dataset/download_cv_bench.py`，做两件事：

1. 用 `snapshot_download` 下载 `nyu-visionx/CV-Bench` 到 `dataset/CV-Bench/`；
2. 对 2D / 3D 子集分别：

   - 读取 `test_2d.parquet` / `test_3d.parquet`；
   - 按 `filename` 去重（`drop_duplicates`）；
   - 用 `random_state=42` 随机抽取 30 张；
   - 保存原始分辨率图片到 `samples/2D`、`samples/3D`；
   - 生成 5 列 × 6 行的总览图 `samples/{2d,3d}_overview.jpg`；
   - 保存 `annotations.csv`，字段为 `image_file`、`task`、`question`、`answer`。

样例数量、随机种子、输出目录都可以通过命令行参数调整，默认值与 Notebook 完全一致。

---

## 2. VSI-Bench

### 数据集简介

VSI-Bench 最适合连续帧、第一视角与空间记忆，与 VLN 高度相关。

它与前两个数据集最大的区别是：输入来自真实室内环境的第一视角连续视频。
模型需要一边理解镜头移动，一边建立室内的空间结构。

例如，摄像头从走廊走进房间，先看到沙发，再转身看见桌子。即使沙发和桌子没有
同时出现在一个画面中，模型也要理解它们之间的空间关系。

它能测出：

- 多帧信息能否整合；
- 物体在不同视角下能否关联；
- 是否记得先前看到的空间结构；
- 能否估计房间尺寸、物体距离和空间布局。

官方提供完整视频、问题标注和评测代码，并有一个用于减少非视觉捷径影响的
VSI-Bench-Debiased 子集。对于 VLN，这比普通的视频事件问答（例如“刚才谁开了门”）
更有针对性。

> 补充（来自数据集自带 README）：仓库提供 `full`、`debiased`、`pruned` 三种配置；
> `pruned` 列标记被 Debiased 人工筛选移除的样本。视频以 `*.zip` 形式随仓库发布。

### 主要任务

评测基于第一视角室内视频，任务围绕空间记忆、视角关联和空间布局估计展开。
官方 README 说明共有 10 种 `question_type`，并分别使用 Accuracy（选择题）
和 MRA（数值题）两类指标；具体指标实现见官方仓库。

### 官方地址

- Hugging Face 数据集：<https://huggingface.co/datasets/nyu-visionx/VSI-Bench>
- 论文（Thinking in Space）：<https://arxiv.org/abs/2412.14171>
- 项目主页：<https://vision-x-nyu.github.io/thinking-in-space.github.io/>
- 代码仓库：<https://github.com/vision-x-nyu/thinking-in-space>

### 当前项目中的数据位置

```text
dataset/VSI-Bench/
├── test.jsonl                  # 全部问题标注
├── test_debiased.parquet       # Debiased 子集标注
├── test_pruned.parquet         # 被筛选掉的样本
├── pruned_ids.txt
├── create_pq.py
├── README.md
└── *.zip                       # 视频文件（arkitscenes / scannet / scannetpp）
```

### 对应处理脚本

`dataset/download_vsi_bench.py`：

- 用 `snapshot_download` 整体下载 `nyu-visionx/VSI-Bench`，包括标注和视频 zip；
- 下载完成后只**检查**关键文件是否存在并打印大小；
- **不解压 zip、不读取视频帧、不生成样例图片或标注**。

原 Notebook 对 VSI-Bench 只执行了 `snapshot_download`，没有 `full` / `debiased` /
`pruned` 的加载代码，也没有视频抽取逻辑，因此脚本没有擅自增加这些步骤。

---

## 3. SPAR-Bench

### 数据集简介

SPAR-Bench 最契合“空间与深度”测试需求。相比 CV-Bench，它更强调真正的三维空间理解。

Notebook 中重点提到：

- 有 RGB 和 RGB-D 两类数据版本；
- RGB-D 版本还提供深度数值、相机位姿及相机内参，能够支持更严格的三维感知实验；
- Tiny 版本只包含 1,000 道题，每种任务 50 道，适合先快速比较模型。

> 注意：普通 VLM 往往只能接收深度图的可视化图像，而不能直接理解深度矩阵或相机内参。
> 使用 RGB-D 版需要设计一致的输入方式。

### 主要任务

Notebook 中列出的能力类别：

- 深度预测（Depth）
- 距离预测（Distance）
- 摄像机运动推理（Camera Motion）
- 多视角位置匹配（Position Matching）
- 视角变化推理（View Change）
- 三维空间想象（Spatial Imagination）

数据集自带 README 说明这些能力覆盖单视角、多视角和视频三类输入场景。

### 官方地址

- Hugging Face 数据集（完整 RGB 版）：<https://huggingface.co/datasets/jasonzhango/SPAR-Bench>
- 论文（From Flatland to Space）：<https://arxiv.org/abs/2503.22976>
- 项目主页：<https://fudan-zvg.github.io/spar>
- 代码仓库：<https://github.com/fudan-zvg/spar>

其他版本（Tiny / RGB-D）的仓库 ID 见数据集 README：
`SPAR-Bench-RGBD`、`SPAR-Bench-Tiny`、`SPAR-Bench-Tiny-RGBD`。

### 当前项目中的数据位置

```text
dataset/SPAR-Bench/
├── README.md
└── data/
    └── test-*.parquet          # 完整 RGB 版本的 Parquet 分片
```

### 对应处理脚本

`dataset/download_spar_bench.py`：

- 用 `snapshot_download` 整体下载 `jasonzhango/SPAR-Bench` 到 `dataset/SPAR-Bench/`；
- 下载完成后只检查 `README.md` 和 `data/` 下的 Parquet 分片；
- **不解析 Parquet、不解码 `image` 字段、不按任务类别抽样、不保存图片或标注**。

> 注意版本差异：Notebook 的文字介绍提到了 Tiny 与 RGB-D 版本，但**实际代码下载的是
> 完整 RGB 版** `jasonzhango/SPAR-Bench`。脚本按实际代码保留该仓库，没有替换成 Tiny。
> 如果确实需要 Tiny 或 RGB-D 版本，请在脚本中修改 `HF_REPO_ID`，并注意字段结构可能不同。

原 Notebook 对 SPAR-Bench 没有本地处理代码，因此上述抽样、多图保存、总览图等功能
目前并不存在，脚本也没有凭空实现。

---

## 4. 使用说明

### 如何运行各数据集脚本

先激活项目 Conda 环境（环境名 `vlm-test`，搭建方式见根目录 `ENV_SETUP.md`），
然后在项目根目录执行：

```bash
# 下载 CV-Bench 并提取 2D/3D 样例
micromamba run -n vlm-test python dataset/download_cv_bench.py

# 下载 SPAR-Bench（完整 RGB 版）
micromamba run -n vlm-test python dataset/download_spar_bench.py

# 下载 VSI-Bench（标注 + 视频 zip）
micromamba run -n vlm-test python dataset/download_vsi_bench.py
```

也可以先 `micromamba activate vlm-test`，再直接用 `python dataset/xxx.py` 运行。
三个脚本都根据**脚本自身位置**计算默认输出目录，不依赖当前工作目录。

常用参数：

| 参数 | 适用脚本 | 说明 |
| --- | --- | --- |
| `--output-dir` | 全部 | 自定义数据目录 |
| `--skip-download` | 全部 | 跳过下载，使用本地已有文件 |
| `--hf-endpoint` | 全部 | 默认 `https://hf-mirror.com`，传 `""` 使用官方地址 |
| `--force` | CV-Bench | 已有样例也重新提取 |
| `--samples` / `--seed` | CV-Bench | 样例数量 / 随机种子，默认 30 / 42 |
| `--max-workers` | SPAR / VSI | 并发下载线程数，默认 4 |

### 依赖说明

三个脚本只使用项目环境中已有的依赖，无需额外安装：

- `huggingface_hub`：下载数据集；
- `pandas`、`pyarrow`：CV-Bench 读取 Parquet；
- `Pillow`：CV-Bench 图片还原、缩略图和总览图。

这些包都已列在根目录 `requirements-vlm-test.txt` 中。脚本不会修改或升级 Conda 环境。

### 输出目录说明

| 脚本 | 默认输出 | 说明 |
| --- | --- | --- |
| `download_cv_bench.py` | `dataset/CV-Bench/` | parquet 与 `samples/`（图片、总览图、annotations.csv） |
| `download_spar_bench.py` | `dataset/SPAR-Bench/` | 仓库原样结构，Parquet 位于 `data/` |
| `download_vsi_bench.py` | `dataset/VSI-Bench/` | 标注文件与视频 zip，保持仓库原样结构 |

### 重复执行与断点续传

- `snapshot_download` 会校验本地文件，已经完整的文件不会重复下载；
- CV-Bench 样例默认在 `samples/` 已存在时跳过，只有加 `--force` 才会覆盖；
- 脚本不会自动删除已有数据或缓存；
- 当前 Git 仓库只跟踪了 README、标注等小文件，Parquet 和视频 zip 等大文件未纳入
  跟踪；如果本地缺失，运行对应脚本会重新下载。

### 关于原 Notebook

原 Notebook `dataset/加载&下载数据集.ipynb` 保留不动，仍可作为参考。
拆分后的脚本与 Notebook 的数据处理逻辑一致，但没有执行过任何下载或样例提取。

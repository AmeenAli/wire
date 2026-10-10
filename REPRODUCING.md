# Reproducing WIRE

Every command runs from the root of the repository. Evaluations write to `results/`, training to `checkpoints/`.

## Environment

The paper's numbers come from Python 3.10, CUDA 13.0 and PyTorch 2.13 on NVIDIA B200 GPUs. The `paper` extra pins the main packages of that environment:

```bash
conda create -n wire python=3.10 -y && conda activate wire
pip install torch==2.13.0 torchvision==0.28.0          # match your CUDA
pip install --no-build-isolation -e ".[paper]"         # builds mamba-ssm and causal-conv1d
```

Keep `transformers` and `qwen-vl-utils` at these versions, since both change frame sampling and resizing between releases. One 80 GB GPU per process is enough for every evaluation. Training runs on any number of GPUs that divides 60, each with at least 80 GB; the paper reports 6.

## CPU check and tests

`python reproduce/cpu_check.py` needs only PyTorch and rebuilds, in a few seconds:

- the input length at 256 frames after contraction: from the paper's 15,394 positions, 14,336 of them visual, to 5,154 with 16 tokens per frame and 2,594 with 6 (Sections 3.1 and 3.3, Appendix D);
- the adapted layers 3, 7, ..., 35 and their 63 maps (Section 3.4);
- the trained parameters: 763M in the hypernetworks and 16.8M in the cross-attention readout, and 203M in the compressor when the Mamba2 kernels are installed (Section 4);
- a generated update of zero at initialization, and merged maps equal to the adapted maps of prefill (Sections 3.4 and 3.5).

After `pip install pytest`, `pytest` runs the tests on a CPU. Among them is WIRE on a tiny, randomly initialized Qwen3-VL with a stand-in for the Mamba2 scan.

## Data

### Evaluation

| Benchmark | How to get it |
| --- | --- |
| Video-MME | `bash eval/videomme/download.sh data/videomme` (900 videos, about 100 GB); the questions are read from the Hub |
| NextQA, PerceptionTest, MVBench, TOMATO, Video-MME with subtitles, LongVideoBench, MLVU | downloaded by lmms-eval on first use (set `HF_HOME`); LongVideoBench and MLVU are gated on the Hub, so request access first |
| LVBench | with `HF_HOME` set, `hf download lmms-lab/LVBench --repo-type dataset --local-dir $HF_HOME/lvbench`, then unpack the videos so that `$HF_HOME/lvbench/<key>.mp4` exists (or set `LVBENCH_DIR` to another folder) |

The other eight benchmarks run through lmms-eval 0.4 with a patch for MLVU, whose answer extraction scores answers such as `D.` as wrong:

```bash
git clone --branch v0.4 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && git apply ../eval/lmms/patches/mlvu_answer_letter.patch && pip install -e . && cd -
```

### Training

The training set is the question-answer part of [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K), without the captions. It keeps the first turn of every video in the open-ended and multiple-choice files. The multiple-choice records appear three times, 1,648,389 records in all.

```bash
hf download lmms-lab/LLaVA-Video-178K --repo-type dataset --local-dir data/LLaVA-Video-178K
# unpack the video archives of every bucket in place, then:
python scripts/data/build_llavavideo.py --data-root data/LLaVA-Video-178K \
    --out data/llavavideo/llavavideo_mcqboost.jsonl
```

For the record order of the paper's runs, use the released `llavavideo_mcqboost.jsonl` (`<HF_LINK_TRAIN_JSONL>`). Its video paths are relative, so set `VIDEO_ROOT=data/LLaVA-Video-178K`.

## From the paper to the commands

| Paper | What it measures | Command |
| --- | --- | --- |
| Table 1 | the released models on nine video benchmarks | `bash reproduce/main_results.sh` |
| Section 5.1 | training-free bilinear pooling with the same contraction, and the uncompressed backbone on short videos | `STAGES="pooling backbone" bash reproduce/ablations.sh` |
| Appendix E | the model with its generated update switched off | `STAGES=update bash reproduce/ablations.sh` |
| Section 4 | training the released models | `NGPU=6 bash scripts/train.sh 16`, and 12, 8 or 6 |

`DRY_RUN=1` prints the commands of any of these scripts without running them. In the reproduce scripts, `STAGES` selects the parts and `MODELS` the models. All evaluations use 256 frames sampled uniformly over the whole clip and greedy decoding (`wire/inference.py`).

### Table 1

The paper reports these accuracies for the released models at 256 frames:

| Model | NextQA | PerceptionTest | MVBench | TOMATO | Video-MME | Video-MME (subtitles) | LongVideoBench | MLVU | LVBench | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `wire-k16` | 81.8 | 66.7 | 62.8 | 30.9 | 66.2 | 67.5 | 60.4 | 52.0 | 44.0 | 59.1 |
| `wire-k12` | 79.6 | 66.2 | 60.9 | 29.3 | 64.1 | 66.5 | 59.6 | 52.5 | 42.0 | 57.9 |
| `wire-k8` | 80.0 | 65.4 | 60.6 | 28.7 | 63.8 | 66.3 | 58.8 | 51.9 | 40.0 | 57.3 |
| `wire-k6` | 79.9 | 63.4 | 60.0 | 28.4 | 62.0 | 65.5 | 58.6 | 50.1 | 41.3 | 56.6 |

Video-MME without subtitles runs through the repository's own harness (`eval/videomme/run.py`), which resumes interrupted runs and splits the questions over several GPUs with `--num-shards` and `--shard-id`. `python eval/videomme/run.py --report-only --output-dir <dir>` merges the shards. The other eight run through lmms-eval with the WIRE plugin (`eval/lmms/run_lmms.sh <model> <task>`). After each run, `eval/lmms/check_format.py` flags runs where more than 2 percent of the answers are not an option letter. For those answers, lmms-eval picks an option at random. `scripts/paper/rescore_mlvu.py` rescores MLVU runs made without the patch.

### Section 5.1 and Appendix E

On Video-MME without subtitles:

| Run | Paper |
| --- | --- |
| `wire-k16`, all videos (from `reproduce/main_results.sh`) | 66.22 |
| Training-free bilinear pooling to 16 tokens per frame (`pool16`) | 62.93 |
| The uncompressed backbone, short videos (`stock`) | 80.67, against 75.56 for `wire-k16` |
| `wire-k16` with its generated update switched off | 1.44 points below `wire-k16` |

## Training

```bash
NGPU=6 bash scripts/train.sh 16                     # also 12, 8 or 6
python scripts/export_release.py checkpoints/wire-k16/checkpoint-5400 \
    release/WIRE-Qwen3-VL-8B-k16 --model wire-k16    # wire.safetensors and wire_config.json
python eval/videomme/run.py --model release/WIRE-Qwen3-VL-8B-k16 \
    --output-dir results/videomme/my-k16
```

The settings are the defaults of `scripts/train.sh` and `wire/train/train.py`:

- AdamW with betas 0.9 and 0.95 and weight decay 0.1;
- a peak learning rate of 3e-5, with 3 percent warm-up and cosine decay;
- a global batch of 120;
- bfloat16 with activation checkpointing, and at most 16,384 tokens per sample.

The loss is the KL divergence to the same frozen backbone reading the full input, on the answer tokens, plus 0.1 times the cross entropy. The global batch is the number of GPUs times 2 times the gradient accumulation, so the number of GPUs must divide 60. Each released model stops at a step of the one-epoch cosine schedule: 5,400 for `wire-k16`, 4,700 for `wire-k12` and 5,500 for `wire-k8` and `wire-k6`. On 6 B200 GPUs a step takes about 42 s with about 50 GB per GPU. Checkpoints hold the WIRE weights, the compressor in bfloat16 and the rest in float32, and the optimizer state. `--model checkpoints/wire-k16/checkpoint-<step>` evaluates one directly.

## What you provide

- The benchmark videos and the training videos listed above, and access to the gated benchmarks on the Hugging Face Hub.
- GPUs with at least 80 GB of memory: one for each evaluation process, and for training a number that divides 60.

## What the release does not contain

- The input-dropout model of Section 5.4 and Table 2. The Video-MME harness holds part of its evaluation (`--condition zero_context`, `frame_drop` and `blind`), and these conditions apply to any WIRE checkpoint.
- The timing runs of Table 4 and Figure 1.
- The runs of BIMBA, Vamba and VideoNSA.

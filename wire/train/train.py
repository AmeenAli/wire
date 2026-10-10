r"""Train WIRE on a frozen Qwen3-VL-8B-Instruct.

Launch with torchrun, e.g. on 4 GPUs (global batch 120 = 4 x 2 x 15):

    torchrun --nproc_per_node 4 -m wire.train.train \
        --data data/llavavideo/llavavideo_mcqboost.jsonl \
        --output-dir checkpoints/wire-k16 --k 16

See scripts/train.sh for the settings and stop step of every released model. Only
the WIRE modules are trained and saved (`wire.safetensors`, ~4 GB in
float32); checkpoints also hold the optimizer state for resuming.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import torch
from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer,
                          TrainerCallback, TrainingArguments)

from ..config import WireConfig
from ..loading import WEIGHTS_NAME, load_wire_weights
from ..modeling import attach_wire
from .data import VideoQADataset, build_collator
from .loss import DistillConfig, DistillLoss


class StopAtStep(TrainerCallback):
    """The released models were stopped early on a schedule planned for the
    full epoch (13,737 steps): keep the schedule, stop and save at N."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step and state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


class WireTrainer(Trainer):
    def __init__(self, *args, distill: DistillLoss, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill = distill
        # the loss is normalized by the global answer-token count
        self.model_accepts_loss_kwargs = True

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        loss = self.distill.compute(self, model, inputs, num_items_in_batch)
        # DDP averages gradients over ranks; scale back so the gradient is
        # the mean over all answer tokens of the global batch
        loss = loss * self.accelerator.num_processes
        return (loss, {"loss": loss}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            logs.update(self.distill.pop_stats())
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        from safetensors.torch import save_file
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        net = self.accelerator.unwrap_model(self.model)
        sd = {k: v.detach().contiguous().cpu()
              for k, v in net.wire.state_dict().items()}
        save_file(sd, os.path.join(output_dir, WEIGHTS_NAME))
        net.wire_state.cfg.save(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        net = self.accelerator.unwrap_model(model or self.model)
        load_wire_weights(net, resume_from_checkpoint)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training jsonl")
    ap.add_argument("--video-root", default=None,
                    help="prefix for relative video paths in --data")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    # architecture
    ap.add_argument("--k", type=int, default=16, help="tokens kept per frame")
    ap.add_argument("--zero-context", action="store_true",
                    help="k = 0 model: no visual token in the decoder input")
    ap.add_argument("--cond-on-queries", action="store_true",
                    help="input-dropout model: condition on the refined queries")
    # objective
    ap.add_argument("--ce-weight", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--context-dropout", type=float, default=0.0)
    ap.add_argument("--frame-drop-p", type=float, default=0.0)
    ap.add_argument("--swap-coef", type=float, default=0.0)
    ap.add_argument("--swap-margin", type=float, default=0.2)
    ap.add_argument("--swap-p", type=float, default=0.5)
    ap.add_argument("--per-rank-dropout-seed", action="store_true")
    # optimization
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--global-batch", type=int, default=120)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--max-grad-norm", type=float, default=2.0)
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help="stop (and save) at this step; the schedule still "
                         "spans the whole epoch")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--resume", default=None,
                    help="checkpoint directory to resume from, or 'last'")
    ap.add_argument("--init-from", default=None,
                    help="start from these WIRE weights with a fresh optimizer")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none")
    return ap.parse_args()


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if args.global_batch % (args.micro_batch * world):
        raise ValueError(f"global batch {args.global_batch} is not divisible "
                         f"by micro batch x world size ({args.micro_batch} x {world})")
    accum = args.global_batch // (args.micro_batch * world)

    cfg = WireConfig(tokens_kept_per_frame=args.k, zero_context=args.zero_context,
                     cond_on_queries=args.cond_on_queries,
                     base_model=args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa")
    attach_wire(model, cfg)
    if args.init_from:
        load_wire_weights(model, args.init_from)
    processor = AutoProcessor.from_pretrained(args.base_model)

    distill = DistillLoss(DistillConfig(
        tau=args.tau, ce_weight=args.ce_weight,
        context_dropout=args.context_dropout, frame_drop_p=args.frame_drop_p,
        swap_coef=args.swap_coef, swap_margin=args.swap_margin,
        swap_p=args.swap_p, per_rank_seed=args.per_rank_dropout_seed),
        rank=rank)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
        optim="adamw_torch_fused",
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        average_tokens_across_devices=True,
        ddp_find_unused_parameters=False,
        ddp_timeout=18000000,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        dataloader_num_workers=args.workers,
        dataloader_prefetch_factor=4 if args.workers else None,
        remove_unused_columns=False,
        seed=args.seed, data_seed=args.seed,
        report_to=args.report_to,
    )
    trainer = WireTrainer(
        model=model, args=targs,
        train_dataset=VideoQADataset(args.data, processor, args.max_length,
                                     seed=args.seed, video_root=args.video_root),
        data_collator=build_collator(processor),
        callbacks=[StopAtStep(args.stop_at_step)],
        distill=distill,
    )
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump({**vars(args), "grad_accum": accum, "world_size": world,
                       "wire_config": cfg.to_dict(),
                       "distill": dataclasses.asdict(distill.cfg)}, f, indent=2)
    resume = args.resume
    if resume == "last":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

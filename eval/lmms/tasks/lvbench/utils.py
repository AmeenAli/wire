import os
import re
import sys
from collections import defaultdict

from loguru import logger as eval_logger

# Parquet columns (lmms-eval/LVBench): video_path, uid, question (question text with
# "(A) ...\n(B) ..." options already inlined), question_type (list[str]), answer (letter),
# time_reference, type (video genre), key (video id = mp4 basename).

hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
cache_dir = os.environ.get("LVBENCH_DIR", os.path.join(hf_home, "lvbench"))

QUESTION_TYPES = ["entity recognition", "event understanding", "key information retrieval", "temporal grounding", "reasoning", "summarization"]
VIDEO_TYPES = ["selfmedia", "cartoon", "live", "tv", "sport", "documentary"]


def lvbench_doc_to_visual(doc):
    video_path = os.path.join(cache_dir, doc["video_path"])
    if not os.path.exists(video_path):
        alt = os.path.join(cache_dir, doc["key"] + ".mp4")
        if os.path.exists(alt):
            video_path = alt
        else:
            sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def lvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "\nAnswer with the option's letter from the given choices directly.")
    question = doc["question"].strip()  # options are already inlined as "(A) ...\n(B) ..."
    return f"{pre_prompt}{question}{post_prompt}"


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def lvbench_process_results(doc, results):
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    qtypes = doc["question_type"]
    if isinstance(qtypes, str):
        qtypes = [qtypes]
    data_dict = {
        "uid": doc["uid"],
        "key": doc["key"],
        "question_type": list(qtypes),
        "video_type": doc["type"],
        "pred_answer": pred_ans,
        "answer": doc["answer"].strip(),
    }
    return {"lvbench_accuracy": data_dict}


def lvbench_aggregate_results(results):
    qtype_score = defaultdict(lambda: {"correct": 0, "answered": 0})
    vtype_score = defaultdict(lambda: {"correct": 0, "answered": 0})
    total_correct, total_answered = 0, 0
    for r in results:
        correct = int(r["pred_answer"] == r["answer"])
        total_answered += 1
        total_correct += correct
        for qt in r["question_type"]:
            qtype_score[qt]["answered"] += 1
            qtype_score[qt]["correct"] += correct
        vtype_score[r["video_type"]]["answered"] += 1
        vtype_score[r["video_type"]]["correct"] += correct

    for qt in QUESTION_TYPES + sorted(set(qtype_score) - set(QUESTION_TYPES)):
        v = qtype_score.get(qt, {"correct": 0, "answered": 0})
        eval_logger.info(f"Evaluation on Question Type: {qt}: {100 * v['correct'] / v['answered'] if v['answered'] > 0 else 0:.2f}% ({v['answered']})")
    for vt in VIDEO_TYPES + sorted(set(vtype_score) - set(VIDEO_TYPES)):
        v = vtype_score.get(vt, {"correct": 0, "answered": 0})
        eval_logger.info(f"Evaluation on Video Type: {vt}: {100 * v['correct'] / v['answered'] if v['answered'] > 0 else 0:.2f}% ({v['answered']})")
    overall = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"Overall Performance: {overall:.2f}% ({total_answered})")
    return overall

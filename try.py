
import os
import re
import glob
import json
import math
import shutil
import argparse
from collections import defaultdict, OrderedDict

import wandb
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2


DEFAULT_ROOT = "/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/outputs/wandb_logs/wandb"


STEP_KEY_PRIORITY = [
    "global_step",
    "trainer/global_step",
    "train/global_step",
    "actor/global_step",
    "critic/global_step",
    "step",
    "_step",
]


def safe_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\/\\:\*\?\"<>\|\s]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "unnamed_run"


def parse_offline_dir_name(path):
    base = os.path.basename(path.rstrip("/"))
    m = re.match(r"^offline-run-(\d{8}_\d{6})-(.+)$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_json_value(value_json):
    if value_json is None:
        return None

    if isinstance(value_json, bytes):
        value_json = value_json.decode("utf-8", errors="ignore")

    try:
        return json.loads(value_json)
    except Exception:
        return None


def is_loggable_number(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def get_history_key(item):
    nested_key = list(getattr(item, "nested_key", []))
    if nested_key:
        return "/".join(nested_key)

    key = getattr(item, "key", "")
    return key


def read_wandb_file(wandb_file):
    """
    返回：
    run_info: dict
    rows: list[dict]
    """
    run_info = {
        "run_id": None,
        "run_name": None,
        "project": None,
        "entity": None,
    }

    rows = []

    ds = DataStore()
    ds.open_for_scan(wandb_file)

    while True:
        data = ds.scan_data()
        if data is None:
            break

        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)

        if rec.HasField("run"):
            run = rec.run
            rid = getattr(run, "run_id", "") or None
            name = getattr(run, "display_name", "") or None
            project = getattr(run, "project", "") or None
            entity = getattr(run, "entity", "") or None

            if rid:
                run_info["run_id"] = rid
            if name:
                run_info["run_name"] = name
            if project:
                run_info["project"] = project
            if entity:
                run_info["entity"] = entity

        if not rec.HasField("history"):
            continue

        row = {}

        for item in rec.history.item:
            key = get_history_key(item)
            if not key:
                continue

            value = parse_json_value(getattr(item, "value_json", None))
            if value is None:
                continue

            row[key] = value

        if row:
            rows.append(row)

    return run_info, rows


def choose_step_for_rows(rows):
    """
    优先使用 global_step / trainer/global_step 等字段。
    如果都没有，就使用 history row 的序号。
    """
    for key in STEP_KEY_PRIORITY:
        vals = []
        for row in rows:
            v = row.get(key)
            if is_loggable_number(v):
                vals.append(int(v))

        if vals:
            return key, vals

    return "__row_index__", list(range(len(rows)))


def aggregate_rows_by_step(rows, steps):
    """
    将同一个 run 内相同 step 的多条 history 合并。
    只保留数值指标，方便重建曲线。
    """
    by_step = OrderedDict()

    for row, step in zip(rows, steps):
        step = int(step)

        if step not in by_step:
            by_step[step] = {}

        for k, v in row.items():
            if k.startswith("_"):
                continue
            if is_loggable_number(v):
                by_step[step][k] = v

        by_step[step]["step"] = step

    return by_step


def read_one_run_dir(run_dir):
    timestamp, dir_run_id = parse_offline_dir_name(run_dir)

    wandb_files = glob.glob(os.path.join(run_dir, "run-*.wandb"))
    if not wandb_files:
        return None

    wandb_file = wandb_files[0]
    file_run_id = None

    m = re.match(r"run-(.+)\.wandb$", os.path.basename(wandb_file))
    if m:
        file_run_id = m.group(1)

    run_info, rows = read_wandb_file(wandb_file)

    final_run_id = run_info.get("run_id") or file_run_id or dir_run_id
    run_name = run_info.get("run_name") or final_run_id

    step_key, steps = choose_step_for_rows(rows)
    unique_steps = sorted(set(int(x) for x in steps))

    return {
        "dir": run_dir,
        "base": os.path.basename(run_dir),
        "timestamp": timestamp,
        "dir_run_id": dir_run_id,
        "file_run_id": file_run_id,
        "run_id": final_run_id,
        "run_name": run_name,
        "project": run_info.get("project"),
        "entity": run_info.get("entity"),
        "wandb_file": wandb_file,
        "rows": rows,
        "step_key": step_key,
        "steps": steps,
        "unique_step_count": len(unique_steps),
        "min_step": min(unique_steps) if unique_steps else None,
        "max_step": max(unique_steps) if unique_steps else None,
    }


def make_output_run(
    output_parent,
    run_name,
    run_id,
    project,
    merged_by_step,
    use_original_id=False,
):
    safe_name = safe_filename(run_name)
    out_dir = os.path.join(output_parent, safe_name)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    merged_run_id = run_id if use_original_id else f"merged_{safe_filename(run_id)}"

    os.environ["WANDB_MODE"] = "offline"

    with wandb.init(
        project=project or "merged-wandb",
        name=run_name,
        id=merged_run_id,
        resume="never",
        dir=out_dir,
    ) as run:
        wandb.define_metric("merged_step")
        wandb.define_metric("*", step_metric="merged_step")

        for step in sorted(merged_by_step.keys()):
            row = dict(merged_by_step[step])
            row["merged_step"] = step
            wandb.log(row, step=step)

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--min-steps", type=int, default=15)
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--use-original-id",
        action="store_true",
        help="默认生成 merged_xxx 作为新 run id；加这个参数会使用原始 run_id。",
    )
    args = parser.parse_args()

    root = args.root.rstrip("/")
    output_parent = root

    offline_dirs = sorted(glob.glob(os.path.join(root, "offline-run-*")))

    print(f"[INFO] root = {root}")
    print(f"[INFO] found offline runs = {len(offline_dirs)}")
    print()

    all_runs = []
    useless = []

    for d in offline_dirs:
        try:
            info = read_one_run_dir(d)
        except Exception as e:
            print(f"[WARN] failed to read {d}: {repr(e)}")
            continue

        if info is None:
            print(f"[WARN] no run-*.wandb found: {d}")
            continue

        all_runs.append(info)

    groups = defaultdict(list)
    for info in all_runs:
        groups[info["run_id"]].append(info)

    for run_id in sorted(groups.keys()):
        runs = sorted(groups[run_id], key=lambda x: x["timestamp"] or "")

        print("=" * 120)
        print(f"[RUN_ID] {run_id}")
        print(f"[RUN_NAME] {runs[0]['run_name']}")
        print(f"[NUM_DIRS] {len(runs)}")

        kept = []

        for r in runs:
            time_runid = f"{r['timestamp']}-{r['run_id']}"
            print(
                f"  {time_runid} | "
                f"unique_steps={r['unique_step_count']} | "
                f"step_key={r['step_key']} | "
                f"range={r['min_step']}->{r['max_step']} | "
                f"dir={r['base']}"
            )

            if r["unique_step_count"] < args.min_steps:
                useless.append(time_runid)
            else:
                kept.append(r)

        if not kept:
            print("  [SKIP GROUP] no valid run after filtering")
            continue

        merged_by_step = OrderedDict()

        for idx, r in enumerate(kept):
            per_run_by_step = aggregate_rows_by_step(r["rows"], r["steps"])

            if not per_run_by_step:
                continue

            incoming_steps = set(per_run_by_step.keys())
            old_steps = set(merged_by_step.keys())
            overlap = old_steps & incoming_steps

            if overlap:
                print(
                    f"  [MERGE] {r['timestamp']}-{r['run_id']} "
                    f"overwrites overlap steps: "
                    f"{min(overlap)}->{max(overlap)}, count={len(overlap)}"
                )
            else:
                print(
                    f"  [MERGE] {r['timestamp']}-{r['run_id']} "
                    f"no overlap"
                )

            for step in sorted(per_run_by_step.keys()):
                # 关键逻辑：后面的 run 覆盖前面相同 step 的记录
                merged_by_step[step] = per_run_by_step[step]

        final_steps = sorted(merged_by_step.keys())
        project = args.project or kept[0].get("project") or "merged-wandb"

        out_dir = make_output_run(
            output_parent=output_parent,
            run_name=kept[0]["run_name"],
            run_id=run_id,
            project=project,
            merged_by_step=merged_by_step,
            use_original_id=args.use_original_id,
        )

        print(
            f"  [DONE] merged_steps={len(final_steps)} | "
            f"range={final_steps[0]}->{final_steps[-1]} | "
            f"output_folder={out_dir}"
        )

    print()
    print("=" * 120)
    print(f"[USELESS] unique_steps < {args.min_steps}")
    if useless:
        for x in useless:
            print(x)
    else:
        print("None")

    useless_path = os.path.join(root, "useless_runs.txt")
    with open(useless_path, "w") as f:
        for x in useless:
            f.write(x + "\n")

    print()
    print(f"[INFO] useless list saved to: {useless_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

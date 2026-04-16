from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def convert_record(record: dict[str, Any], row_index: int, image_folder: str) -> dict[str, str]:
    answers = record["answers"]
    image_name = Path(record["image_path"]).name
    example_id = record.get("example_id") or Path(image_name).stem or f"example_{row_index:05d}"

    return {
        "example_id": example_id,
        "prompt": record["prompt"],
        "image_rel_path": f"{image_folder.rstrip('/')}/{image_name}",
        "q1_hit_object": str(answers["q1_hit_object"]),
        "q2_visible_objects": json.dumps(answers["q2_visible_objects"], ensure_ascii=False),
        "q3a_visible_overlapping_objects": json.dumps(answers["q3a_visible_overlapping_objects"], ensure_ascii=False),
        "q3b_layer_groups_bottom_to_top": json.dumps(answers["q3b_layer_groups_bottom_to_top"], ensure_ascii=False),
    }


def convert_jsonl_to_csv(input_jsonl_path: str | Path, output_csv_path: str | Path, image_folder: str = "image") -> Path:
    input_jsonl_path = Path(input_jsonl_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "example_id",
        "prompt",
        "image_rel_path",
        "q1_hit_object",
        "q2_visible_objects",
        "q3a_visible_overlapping_objects",
        "q3b_layer_groups_bottom_to_top",
    ]

    with input_jsonl_path.open("r", encoding="utf-8") as src, output_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, line in enumerate(src):
            if not line.strip():
                continue
            record = json.loads(line)
            writer.writerow(convert_record(record, row_index=row_index, image_folder=image_folder))

    return output_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the generator JSONL into a flat CSV for a Kaggle dataset. "
            "The CSV stores relative image paths such as image/example_00001_question.png."
        )
    )
    parser.add_argument("input_jsonl", help="Path to the generated dataset.jsonl")
    parser.add_argument("output_csv", help="Where to write the Kaggle-ready CSV")
    parser.add_argument(
        "--image-folder", default="image",
        help="Relative image folder inside the Kaggle dataset. Default: image",
    )
    args = parser.parse_args()

    output_path = convert_jsonl_to_csv(args.input_jsonl, args.output_csv, image_folder=args.image_folder)
    print(output_path)


if __name__ == "__main__":
    main()

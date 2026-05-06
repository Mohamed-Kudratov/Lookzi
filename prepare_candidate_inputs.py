"""
Export Lookzi benchmark pairs into a simple folder for external VTON engines.

Colab usage:
  python prepare_candidate_inputs.py \
    --pairs benchmark/catalog_pairs.json \
    --output_dir /content/drive/MyDrive/Lookzi/candidate_inputs
"""

import argparse
import json
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Export Lookzi benchmark inputs")
    parser.add_argument("--pairs", default="benchmark/catalog_pairs.json")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def copy_image(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    with open(args.pairs, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    for pair in data["pairs"]:
        pair_id = pair["id"]
        tag = pair["tag"]
        person_dst = os.path.join(args.output_dir, f"{pair_id}_person.png")
        garment_dst = os.path.join(args.output_dir, f"{pair_id}_garment.png")
        copy_image(pair["person"], person_dst)
        copy_image(pair["garment"], garment_dst)
        manifest.append(
            {
                "id": pair_id,
                "tag": tag,
                "cloth_type": pair["cloth_type"],
                "person": person_dst,
                "garment": garment_dst,
                "expected_output": os.path.join(
                    "/content/drive/MyDrive/Lookzi/idm_vton_outputs",
                    f"{pair_id}.png",
                ),
            }
        )

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[Inputs] Exported {len(manifest)} pairs")
    print(f"[Inputs] Folder: {args.output_dir}")
    print(f"[Inputs] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

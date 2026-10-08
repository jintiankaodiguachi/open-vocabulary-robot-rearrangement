#!/usr/bin/env python3
import argparse
import json
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("prompt")
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8090/api/v1/segment_uploaded",
    )
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        r = requests.post(
            args.url,
            files={"image": ("wrist.jpg", f, "image/jpeg")},
            data={
                "text_prompt": args.prompt,
                "include_mask_rle": "true",
                "save_debug_image": "true",
            },
            timeout=(2.0, 30.0),
        )

    print("HTTP", r.status_code)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

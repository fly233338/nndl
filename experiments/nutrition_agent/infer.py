import argparse
import json

from .graph import build_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to a generated meal JSON file")
    args = parser.parse_args()
    raw = open(args.json, encoding="utf-8").read()
    print(json.dumps(build_graph().invoke({"raw_json": raw}), ensure_ascii=False, indent=2))

#!/usr/bin/env python3
"""Compare the clean manifest's shard hints with actual Parquet episode rows."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq


manifest = json.loads(Path(sys.argv[1]).read_text())
source = Path(sys.argv[2])
expected = {int(row["episode_index"]): row for row in manifest["episodes"]}
shards = defaultdict(set)
for row in expected.values():
    shards[int(row["data_chunk_index"]), int(row["data_file_index"])].add(int(row["episode_index"]))
seen = Counter()
for (chunk, file_index), ids in sorted(shards.items()):
    path = source / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
    table = pq.read_table(path, columns=["episode_index"], filters=[("episode_index", "in", sorted(ids))])
    seen.update(map(int, table["episode_index"].to_numpy()))
missing = sorted(set(expected) - set(seen))
mismatch = sorted((i, int(expected[i]["length"]), seen[i]) for i in expected if seen[i] != int(expected[i]["length"]))
actual = {}
for path in sorted((source / "meta/episodes").rglob("*.parquet")):
    table = pq.read_table(path, columns=["episode_index", "data/chunk_index", "data/file_index"],
                          filters=[("episode_index", "in", missing)])
    for item in table.to_pylist():
        episode_id = int(item["episode_index"])
        actual[episode_id] = [int(item["data/chunk_index"]), int(item["data/file_index"])]
print(json.dumps({"found_episodes": len(seen), "missing": missing, "length_mismatch": mismatch,
                  "source_metadata_shards": actual,
                  "old_manifest_shards": {i: [expected[i]["data_chunk_index"], expected[i]["data_file_index"]]
                                          for i in missing}}, indent=2))

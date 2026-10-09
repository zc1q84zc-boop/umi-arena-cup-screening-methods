#!/usr/bin/env python3
"""Locate clean episodes whose motion crosses a packed Parquet file boundary."""

import json
from pathlib import Path
import sys

import pyarrow.parquet as pq


source = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
wanted = {27111, 78305, 89281, 117931, 144608, 175551, 209759, 255212, 283778, 329865}
for row in manifest["episodes"]:
    episode_id = int(row["episode_index"])
    if episode_id not in wanted:
        continue
    chunk, file_index = int(row["data_chunk_index"]), int(row["data_file_index"])
    found = []
    for offset in (-2, -1, 0, 1, 2):
        if file_index + offset < 0:
            continue
        path = source / f"data/chunk-{chunk:03d}/file-{file_index+offset:03d}.parquet"
        if not path.is_file():
            continue
        table = pq.read_table(path, columns=["episode_index", "frame_index"],
                              filters=[("episode_index", "=", episode_id)])
        if len(table):
            indices = table["frame_index"].to_numpy()
            found.append({"offset": offset, "file": str(path), "rows": len(table),
                          "first_frame": int(indices.min()), "last_frame": int(indices.max())})
    print(json.dumps({"episode": episode_id, "expected_length": row["length"], "found": found}), flush=True)

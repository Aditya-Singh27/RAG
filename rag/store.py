"""LanceDB-backed vector store. Embedded, file-based: no always-on server."""
from __future__ import annotations
import json
from pathlib import Path

import lancedb
import pyarrow as pa


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class VectorStore:
    TABLE = "chunks"

    def __init__(self, uri: str, embed_model: str, dim: int):
        Path(uri).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(uri)
        self.dim, self.embed_model = dim, embed_model
        meta = Path(uri) / "store_meta.json"
        if meta.exists():                      # refuse to mix vector spaces
            m = json.loads(meta.read_text())
            if (m["embed_model"], m["dim"]) != (embed_model, dim):
                raise ValueError(
                    f"store was built with {m['embed_model']}/{m['dim']}d, "
                    f"got {embed_model}/{dim}d. Use a new LANCEDB_URI or re-ingest.")
        else:
            meta.write_text(json.dumps({"embed_model": embed_model, "dim": dim}))
        schema = pa.schema([
            ("chunk_id", pa.string()), ("doc_id", pa.string()), ("source", pa.string()),
            ("doc_type", pa.string()), ("page", pa.int32()), ("chunk_index", pa.int32()),
            ("doc_hash", pa.string()), ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), dim)),
        ])
        try:
            self.tbl = self.db.open_table(self.TABLE)
        except Exception:                      # table does not exist yet
            self.tbl = self.db.create_table(self.TABLE, schema=schema)

    def count(self) -> int:
        return self.tbl.count_rows()

    def doc_hash(self, doc_id: str) -> str | None:
        rows = self.tbl.search().where(f"doc_id = {_q(doc_id)}", prefilter=True).limit(1).select(["doc_hash"]).to_list()
        return rows[0]["doc_hash"] if rows else None

    def upsert(self, rows: list[dict]) -> None:
        if rows:
            (self.tbl.merge_insert("chunk_id").when_matched_update_all()
             .when_not_matched_insert_all().execute(rows))

    def delete_stale(self, doc_id: str, keep_hash: str) -> None:
        self.tbl.delete(f"doc_id = {_q(doc_id)} AND doc_hash != {_q(keep_hash)}")

    def delete_doc(self, doc_id: str) -> None:
        self.tbl.delete(f"doc_id = {_q(doc_id)}")

    def search(self, qvec: list[float], k: int = 5, doc_type: str | None = None,
               doc_id: str | None = None) -> list[dict]:
        q = self.tbl.search(qvec).metric("cosine")
        conds = []
        if doc_type:
            conds.append(f"doc_type = {_q(doc_type)}")
        if doc_id:
            conds.append(f"doc_id = {_q(doc_id)}")
        if conds:
            q = q.where(" AND ".join(conds), prefilter=True)
        out = []
        for r in q.limit(k).to_list():
            r["score"] = 1.0 - r.pop("_distance")      # cosine similarity
            r.pop("vector", None)
            out.append(r)
        return out

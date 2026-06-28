"""MCAP -> columnar (Parquet) conversion (P3, Tier-1 data-platform milestone).

Writes a small MCAP with a JSON topic and a non-JSON topic, then checks ``convert_mcap``
produces one flattened Parquet table per JSON topic and reports the others as skipped.
"""

import json

import pytest

import pyroboframes as prf

# The pure-Python `mcap` writer is only needed to author the test fixture.
mcap_writer = pytest.importorskip("mcap.writer")
pq = pytest.importorskip("pyarrow.parquet")


def _write_mcap(path: str) -> None:
    with open(path, "wb") as fh:
        w = mcap_writer.Writer(fh)
        w.start()

        sid = w.register_schema(name="state", encoding="jsonschema", data=b"")
        json_chan = w.register_channel(topic="/state", message_encoding="json", schema_id=sid)
        for i, state in enumerate([[1.0, 2.0], [3.0, 4.0]]):
            payload = json.dumps({"observation": {"state": state}, "gripper": i == 0})
            w.add_message(
                channel_id=json_chan,
                log_time=(i + 1) * 1000,
                publish_time=(i + 1) * 1000,
                sequence=i,
                data=payload.encode(),
            )

        # A non-JSON topic that must be reported as skipped.
        rid = w.register_schema(name="raw", encoding="protobuf", data=b"")
        raw_chan = w.register_channel(topic="/raw", message_encoding="protobuf", schema_id=rid)
        w.add_message(channel_id=raw_chan, log_time=5000, publish_time=5000, sequence=0, data=b"\xde\xad")

        w.finish()


def test_convert_mcap_flattens_json_topics(tmp_path):
    mcap_path = tmp_path / "run.mcap"
    _write_mcap(str(mcap_path))
    out_dir = tmp_path / "out"

    report = prf.convert_mcap(str(mcap_path), str(out_dir))

    # Exactly one JSON topic converted; the protobuf topic skipped.
    assert report["skipped"] == ["/raw"]
    assert len(report["topics"]) == 1
    t = report["topics"][0]
    assert t["topic"] == "/state"
    assert t["messages"] == 2
    assert t["columns"] == 3  # observation.state.0, observation.state.1, gripper

    # The written Parquet is a real, flattened columnar table.
    table = pq.read_table(t["path"])
    cols = set(table.column_names)
    assert {"log_time", "observation.state.0", "observation.state.1", "gripper"} <= cols
    assert table.num_rows == 2
    d = table.to_pydict()
    assert d["log_time"] == [1000, 2000]
    assert d["observation.state.0"] == [1.0, 3.0]
    assert d["gripper"] == [True, False]

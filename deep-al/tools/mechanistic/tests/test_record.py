import csv
from mechanistic.proxies import ProxySnapshot
from mechanistic.record import RECORD_COLUMNS, write_proxy_record

def test_paper_named_record_is_resume_safe(tmp_path):
    snapshot = ProxySnapshot(0.2, 0.1, 0.2, 0.3, 4.0, 0.5)
    path = tmp_path / "mechanistic_proxy_records.csv"
    write_proxy_record(path, snapshot, episode=2, method="typiclust", seed=1, annotation_budget=20, test_accuracy=80)
    write_proxy_record(path, snapshot, episode=2, method="typiclust", seed=1, annotation_budget=20, test_accuracy=81)
    with path.open(newline="") as handle: rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == RECORD_COLUMNS
    assert len(rows) == 1 and rows[0]["test_accuracy"] == "81.0"

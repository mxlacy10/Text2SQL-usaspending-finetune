"""Download USAspending prime award data via the public bulk-download API.

API reference (verified against the live endpoint):
  POST https://api.usaspending.gov/api/v2/bulk_download/awards/
    body: {"filters": {"prime_award_types": [...], "date_type": "action_date",
                        "date_range": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}}
    -> {"status_url": ..., "file_name": ..., "file_url": ...}

  GET  <status_url>
    -> {"status": "running" | "finished" | "failed", "file_url": ..., ...}

No API key is required. One request is issued per fiscal year so a failure
or slow year can be retried independently and raw files land in
data/raw/<fiscal_year>/.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import httpx

from t2s.common.config import AwardType, BaseConfig

BULK_DOWNLOAD_URL = "https://api.usaspending.gov/api/v2/bulk_download/awards/"

# USAspending prime award type codes.
# https://www.usaspending.gov/data-dictionary -> "Award Type"
AWARD_TYPE_CODES: dict[AwardType, list[str]] = {
    "contracts": ["A", "B", "C", "D"],
}

# Federal fiscal years run Oct (year - 1) through Sep (year).
# See AGENTS.md: "Federal fiscal years run Oct-Sep."


def fiscal_year_date_range(fiscal_year: int) -> tuple[str, str]:
    """Return the (start_date, end_date) ISO strings for a federal fiscal year."""
    start = f"{fiscal_year - 1}-10-01"
    end = f"{fiscal_year}-09-30"
    return start, end


def build_filters(
    fiscal_year: int, award_types: list[AwardType], agencies: list[str] | None = None
) -> dict[str, object]:
    """Build the `filters` payload for one fiscal year's bulk-download request.

    `agencies` names toptier awarding agencies to bound the slice (e.g.
    "Department of Veterans Affairs"). None or empty = no agency filter.
    """
    start_date, end_date = fiscal_year_date_range(fiscal_year)
    codes = [code for award_type in award_types for code in AWARD_TYPE_CODES[award_type]]
    filters: dict[str, object] = {
        "prime_award_types": codes,
        "date_type": "action_date",
        "date_range": {"start_date": start_date, "end_date": end_date},
    }
    if agencies:
        filters["agencies"] = [
            {"type": "awarding", "tier": "toptier", "name": name} for name in agencies
        ]
    return filters


def request_bulk_download(filters: dict[str, object], client: httpx.Client) -> dict[str, str]:
    """Submit a bulk-download request and return {status_url, file_name, file_url}."""
    response = client.post(BULK_DOWNLOAD_URL, json={"filters": filters}, timeout=60)
    response.raise_for_status()
    body: dict[str, str] = response.json()
    return body


def poll_until_ready(
    status_url: str,
    client: httpx.Client,
    poll_interval_seconds: float = 10.0,
    timeout_seconds: float = 3600.0,
) -> dict[str, object]:
    """Poll a bulk-download status URL until it finishes, fails, or times out."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = client.get(status_url, timeout=60)
        response.raise_for_status()
        status: dict[str, object] = response.json()
        if status["status"] == "finished":
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"Bulk download failed: {status}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Bulk download did not finish within {timeout_seconds}s: {status}")
        time.sleep(poll_interval_seconds)


def download_and_extract(file_url: str, dest_dir: Path, client: httpx.Client) -> list[Path]:
    """Download a zip file and extract its CSVs into dest_dir. Returns extracted paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / Path(file_url).name
    with client.stream("GET", file_url, timeout=600, follow_redirects=True) as response:
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                zf.extract(name, dest_dir)
                extracted.append(dest_dir / name)
    zip_path.unlink()
    return extracted


def download_fiscal_year(
    fiscal_year: int,
    award_types: list[AwardType],
    raw_dir: Path,
    client: httpx.Client,
    agencies: list[str] | None = None,
    poll_interval_seconds: float = 10.0,
    timeout_seconds: float = 3600.0,
) -> list[Path]:
    """Request, wait for, download, and extract one fiscal year's award data."""
    filters = build_filters(fiscal_year, award_types, agencies)
    submission = request_bulk_download(filters, client)
    status = poll_until_ready(
        submission["status_url"],
        client,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    file_url = str(status["file_url"])
    dest_dir = raw_dir / str(fiscal_year)
    return download_and_extract(file_url, dest_dir, client)


def download_award_data(
    config: BaseConfig,
    poll_interval_seconds: float = 10.0,
    timeout_seconds: float = 3600.0,
) -> dict[int, list[Path]]:
    """Download all configured fiscal years of award data into config.data.raw_dir."""
    raw_dir = Path(config.data.raw_dir)
    results: dict[int, list[Path]] = {}
    with httpx.Client() as client:
        for fiscal_year in config.data.fiscal_years:
            results[fiscal_year] = download_fiscal_year(
                fiscal_year,
                config.data.award_types,
                raw_dir,
                client,
                agencies=config.data.agencies,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
            )
    return results

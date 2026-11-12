"""Client for the hosted Open Reaction Database API (free, no key; run by the ORD project).

Only three calls are used: a search for reaction ids, a fetch of reactions by id, and the list
of datasets (for the grant year). Every failure is raised as `OrdApiError` with a message the
caller turns into a tool error, so nothing reaches the agent loop as an exception.

Measured on 2026-11-13: a product search took 3.6 s (exact) and 9 s (similarity) on a free
server, but the shared server answered 503 "every search slot stayed busy" or 504 for minutes
after heavy queries. 503 is retried; 504 is not (the server keeps searching and holds a slot).
"""

from __future__ import annotations

import re
import time

import requests

RETRY_WAIT_S = 5.0
BUSY_RETRIES = 2
FETCH_BATCH = 100  # the API returns at most 100 reactions per request
_YEAR = re.compile(r"(\d{4})$")


class OrdApiError(RuntimeError):
    """The hosted API did not answer usably; the message says why."""


class OrdClient:
    """Thin, retrying wrapper over the three endpoints."""

    def __init__(
        self,
        base_url: str,
        search_timeout_s: float,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = search_timeout_s
        self._session = session or requests.Session()
        self._sleep = sleep
        self._years: dict[str, int] | None = None

    def search_ids(self, query: dict) -> list[str]:
        """Reaction ids matching `query` (the body of POST /search)."""
        for attempt in range(BUSY_RETRIES + 1):
            response = self._send("POST", "/search", json=query)
            if response.status_code == 503 and attempt < BUSY_RETRIES:
                self._sleep(RETRY_WAIT_S)
                continue
            return self._json(response).get("reaction_ids", [])
        raise OrdApiError("unreachable")  # pragma: no cover

    def reactions(self, ids: list[str]) -> list[dict]:
        """Reaction records (`reaction_id`, `dataset_id`, `binpb`, `molblocks`) by id."""
        records: list[dict] = []
        for start in range(0, len(ids), FETCH_BATCH):
            response = self._send(
                "GET", "/reactions", params={"id": ids[start : start + FETCH_BATCH]}
            )
            records.extend(self._json(response))
        return records

    def dataset_years(self) -> dict[str, int]:
        """Grant year per dataset id, from names such as `uspto-grants-1976` (cached)."""
        if self._years is None:
            datasets = self._json(self._send("GET", "/datasets"))
            self._years = {
                d["dataset_id"]: int(m.group(1))
                for d in datasets
                if (m := _YEAR.search(d.get("name", "")))
            }
        return self._years

    def _send(self, method: str, path: str, **kwargs) -> requests.Response:
        try:
            return self._session.request(
                method, f"{self._base}{path}", timeout=self._timeout, **kwargs
            )
        except requests.Timeout as error:
            raise OrdApiError(
                f"The ORD API did not answer within {self._timeout:.0f} s."
            ) from error
        except requests.RequestException as error:
            raise OrdApiError(
                f"The ORD API could not be reached: {type(error).__name__}."
            ) from error

    @staticmethod
    def _json(response: requests.Response):
        if response.status_code == 503:
            raise OrdApiError("The ORD search service is busy (all search slots in use).")
        if response.status_code == 504:
            raise OrdApiError("The ORD search timed out on the server (it is shared and slow).")
        if response.status_code != 200:
            raise OrdApiError(f"The ORD API answered HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError as error:
            raise OrdApiError("The ORD API returned something that is not JSON.") from error

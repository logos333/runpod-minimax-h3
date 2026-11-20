from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from client.config import ClientSettings, get_client_settings

STATE_FILE = Path(".runpod_state.json")


class RunPodManager:
    """Lifecycle management (status, start, stop/pause, teardown) for RunPod Serverless."""

    def __init__(self, settings: ClientSettings) -> None:
        self._settings = settings
        self._api_key = settings.resolved_api_key()

    def _gql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                self._settings.runpod_graphql_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "minimax-h3-manager/1.0",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        if data.get("errors"):
            raise RuntimeError(f"RunPod GraphQL error: {json.dumps(data['errors'], indent=2)}")
        return data["data"]

    def _find_endpoint(self, acct: dict[str, Any], endpoint_id: str | None) -> dict[str, Any] | None:
        target_id = endpoint_id or self._settings.runpod_endpoint_id
        for ep in acct.get("endpoints") or []:
            if target_id and ep["id"] == target_id:
                return ep
            if not target_id and ep["name"] == self._settings.runpod_endpoint_name:
                return ep
        return None

    def get_account(self) -> dict[str, Any]:
        return self._gql(
            """
            query {
              myself {
                clientBalance
                currentSpendPerHr
                networkVolumes { id name size dataCenterId }
                endpoints { id name gpuIds workersMin workersMax idleTimeout templateId networkVolumeId }
              }
            }
            """
        )["myself"]

    def get_endpoint_health(self, endpoint_id: str) -> dict[str, Any]:
        url = f"{self._settings.runpod_api_base.rstrip('/')}/{endpoint_id}/health"
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, headers={"Authorization": f"Bearer {self._api_key}"})
            resp.raise_for_status()
            return resp.json()

    def status(self, endpoint_id: str | None = None) -> None:
        acct = self.get_account()
        ep = self._find_endpoint(acct, endpoint_id)
        print(f"Balance:          ${acct.get('clientBalance', 0):.4f}")
        print(f"Current Spend/hr: ${acct.get('currentSpendPerHr', 0):.4f}/hr")
        print(f"Network Volumes:  {len(acct.get('networkVolumes') or [])}")
        for v in acct.get("networkVolumes") or []:
            est_mo = v["size"] * 0.07
            print(f"  - {v['id']} | {v['name']} | {v['size']}GB ({v['dataCenterId']}) | ~${est_mo:.2f}/mo")

        if not ep:
            print("Endpoint:         NONE (run `uv run python -m scripts.deploy_runpod` to create)")
            return

        state_label = "PAUSED (0 max workers)" if ep["workersMax"] == 0 else "ACTIVE (auto-scales to 0 when idle)"
        print(f"Endpoint:         {ep['id']} ({ep['name']}) — {state_label}")
        print(f"  GPUs:           {ep['gpuIds']}")
        print(f"  Workers:        min={ep['workersMin']}, max={ep['workersMax']}, idleTimeout={ep['idleTimeout']}s")
        print(f"  Volume:         {ep.get('networkVolumeId')}")
        try:
            health = self.get_endpoint_health(ep["id"])
            print(f"  Live Health:    {json.dumps(health)}")
        except Exception as exc:
            print(f"  Live Health:    unavailable ({exc})")

    def set_workers(
        self,
        workers_min: int,
        workers_max: int,
        endpoint_id: str | None = None,
    ) -> None:
        acct = self.get_account()
        ep = self._find_endpoint(acct, endpoint_id)
        if not ep:
            raise RuntimeError("No RunPod endpoint found. Run `uv run python -m scripts.deploy_runpod` first.")

        ep_input: dict[str, Any] = {
            "id": ep["id"],
            "name": ep["name"],
            "templateId": ep["templateId"],
            "gpuIds": ep["gpuIds"],
            "workersMin": workers_min,
            "workersMax": workers_max,
            "idleTimeout": ep.get("idleTimeout") or self._settings.runpod_idle_timeout_sec,
            "scalerType": "REQUEST_COUNT",
            "scalerValue": 1,
        }
        if ep.get("networkVolumeId"):
            ep_input["networkVolumeId"] = ep["networkVolumeId"]

        res = self._gql(
            """
            mutation($input: EndpointInput!) {
              saveEndpoint(input: $input) { id name workersMin workersMax }
            }
            """,
            {"input": ep_input},
        )["saveEndpoint"]
        print(
            f"[manage] Updated endpoint {res['id']} ({res['name']}): "
            f"workersMin={res['workersMin']}, workersMax={res['workersMax']}"
        )

    def purge_queue(self, endpoint_id: str) -> None:
        url = f"{self._settings.runpod_api_base.rstrip('/')}/{endpoint_id}/purge-queue"
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers={"Authorization": f"Bearer {self._api_key}"})
            if resp.status_code < 400:
                print(f"[manage] Purged pending queue for {endpoint_id}: {resp.text}")

    def teardown(self, endpoint_id: str | None = None, delete_volume: bool = False) -> None:
        acct = self.get_account()
        ep = self._find_endpoint(acct, endpoint_id)
        if ep:
            self.purge_queue(ep["id"])
            self.set_workers(0, 0, endpoint_id=ep["id"])
            self._gql(
                """
                mutation($id: String!) {
                  deleteEndpoint(id: $id)
                }
                """,
                {"id": ep["id"]},
            )
            print(f"[manage] Deleted endpoint {ep['id']} ({ep['name']}).")
        else:
            print("[manage] No matching endpoint to delete.")

        if delete_volume:
            for vol in acct.get("networkVolumes") or []:
                if vol["name"] == self._settings.runpod_volume_name:
                    self._gql(
                        """
                        mutation($input: DeleteNetworkVolumeInput!) {
                          deleteNetworkVolume(input: $input)
                        }
                        """,
                        {"input": {"id": vol["id"]}},
                    )
                    print(f"[manage] Deleted Network Volume {vol['id']} ({vol['name']}). Storage billing is now $0/mo.")
        else:
            print("[manage] Network Volume kept intact (fast redeploy later). Pass --delete-volume to hit $0/mo storage.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage MiniMax-H3 RunPod Serverless lifecycle")
    parser.add_argument("--endpoint-id", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show RunPod balance, active spend, endpoint health, and volume")

    p_start = sub.add_parser("start", help="Enable endpoint (auto-scales 0..max_workers on demand)")
    p_start.add_argument("--min-workers", type=int, default=0, help="Keep 0 for pay-per-use (recommended)")
    p_start.add_argument("--max-workers", type=int, default=3, help="Max parallel GPUs")

    sub.add_parser("stop", help="Pause endpoint (min=0, max=0, purge queue -> $0 GPU billing)")

    p_down = sub.add_parser("teardown", help="Delete endpoint (and optionally the Network Volume)")
    p_down.add_argument("--delete-volume", action="store_true", help="Also delete the 60GB Network Volume ($0/mo)")

    args = parser.parse_args()
    mgr = RunPodManager(get_client_settings())

    if args.cmd == "status":
        mgr.status(endpoint_id=args.endpoint_id)
    elif args.cmd == "start":
        mgr.set_workers(args.min_workers, args.max_workers, endpoint_id=args.endpoint_id)
    elif args.cmd == "stop":
        acct = mgr.get_account()
        ep = mgr._find_endpoint(acct, args.endpoint_id)
        if not ep:
            print("[manage] No endpoint found.", file=sys.stderr)
            sys.exit(1)
        mgr.purge_queue(ep["id"])
        mgr.set_workers(0, 0, endpoint_id=ep["id"])
    elif args.cmd == "teardown":
        mgr.teardown(endpoint_id=args.endpoint_id, delete_volume=args.delete_volume)


if __name__ == "__main__":
    main()

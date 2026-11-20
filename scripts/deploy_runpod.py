from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from client.config import ClientSettings, get_client_settings

STATE_FILE = Path(".runpod_state.json")


class RunPodProvisioner:
    """Idempotent RunPod GraphQL provisioner for Network Volume, Template, and Serverless Endpoint."""

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
                    "User-Agent": "minimax-h3-provisioner/1.0",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        if data.get("errors"):
            raise RuntimeError(f"RunPod GraphQL error: {json.dumps(data['errors'], indent=2)}")
        return data["data"]

    def get_account_state(self) -> dict[str, Any]:
        return self._gql(
            """
            query {
              myself {
                clientBalance
                networkVolumes { id name size dataCenterId }
                endpoints { id name gpuIds workersMin workersMax networkVolumeId templateId }
                podTemplates { id name imageName isServerless }
              }
            }
            """
        )["myself"]

    def ensure_network_volume(self, existing: list[dict[str, Any]]) -> dict[str, Any]:
        for vol in existing:
            if vol["name"] == self._settings.runpod_volume_name:
                print(f"[deploy] Reusing Network Volume: {vol['id']} ({vol['name']}, {vol['size']}GB, {vol['dataCenterId']})")
                return vol

        print(
            f"[deploy] Creating Network Volume '{self._settings.runpod_volume_name}' "
            f"({self._settings.runpod_volume_size_gb}GB in {self._settings.runpod_datacenter_id})..."
        )
        res = self._gql(
            """
            mutation($input: CreateNetworkVolumeInput!) {
              createNetworkVolume(input: $input) {
                id
                name
                size
                dataCenterId
              }
            }
            """,
            {
                "input": {
                    "name": self._settings.runpod_volume_name,
                    "size": self._settings.runpod_volume_size_gb,
                    "dataCenterId": self._settings.runpod_datacenter_id,
                }
            },
        )
        vol = res["createNetworkVolume"]
        print(f"[deploy] Created Network Volume: {vol['id']}")
        return vol

    def ensure_template(self, existing: list[dict[str, Any]]) -> dict[str, Any]:
        template_id = None
        for tpl in existing:
            if tpl["name"] == self._settings.runpod_template_name and tpl.get("isServerless"):
                template_id = tpl["id"]
                break

        env_vars = [
            {"key": "MODELS_ROOT", "value": "/runpod-volume/models"},
            {"key": "AUTO_SYNC_ON_BOOT", "value": "true"},
        ]
        optional_env = {
            "HF_TOKEN": self._settings.hf_token,
            "S3_ENDPOINT_URL": self._settings.s3_endpoint_url,
            "S3_BUCKET": self._settings.s3_bucket,
            "S3_ACCESS_KEY_ID": self._settings.s3_access_key_id,
            "S3_SECRET_ACCESS_KEY": self._settings.s3_secret_access_key,
            "S3_PUBLIC_BASE_URL": self._settings.s3_public_base_url,
        }
        for k, v in optional_env.items():
            if v:
                env_vars.append({"key": k, "value": v})

        template_input: dict[str, Any] = {
            "name": self._settings.runpod_template_name,
            "imageName": self._settings.worker_image,
            "dockerArgs": "",
            "containerDiskInGb": 30,
            "volumeInGb": 0,
            "isServerless": True,
            "env": env_vars,
        }
        if template_id:
            template_input["id"] = template_id

        action = "Updating" if template_id else "Creating"
        print(f"[deploy] {action} Serverless Template '{self._settings.runpod_template_name}' ({self._settings.worker_image})...")
        res = self._gql(
            """
            mutation($input: SaveTemplateInput!) {
              saveTemplate(input: $input) {
                id
                name
                imageName
                isServerless
              }
            }
            """,
            {"input": template_input},
        )
        tpl = res["saveTemplate"]
        print(f"[deploy] Template ready: {tpl['id']}")
        return tpl

    def ensure_endpoint(
        self,
        existing: list[dict[str, Any]],
        template_id: str,
        network_volume_id: str | None,
    ) -> dict[str, Any]:
        endpoint_id = None
        for ep in existing:
            if ep["name"] == self._settings.runpod_endpoint_name:
                endpoint_id = ep["id"]
                break

        ep_input: dict[str, Any] = {
            "name": self._settings.runpod_endpoint_name,
            "templateId": template_id,
            "gpuIds": self._settings.runpod_gpu_ids,
            "workersMin": self._settings.runpod_workers_min,
            "workersMax": self._settings.runpod_workers_max,
            "idleTimeout": self._settings.runpod_idle_timeout_sec,
            "executionTimeoutMs": self._settings.runpod_execution_timeout_ms,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
        if endpoint_id:
            ep_input["id"] = endpoint_id
        if network_volume_id:
            ep_input["networkVolumeId"] = network_volume_id

        action = "Updating" if endpoint_id else "Creating"
        print(
            f"[deploy] {action} Serverless Endpoint '{self._settings.runpod_endpoint_name}' "
            f"(gpuIds={self._settings.runpod_gpu_ids}, vol={network_volume_id})..."
        )
        res = self._gql(
            """
            mutation($input: EndpointInput!) {
              saveEndpoint(input: $input) {
                id
                name
                gpuIds
                templateId
                networkVolumeId
                workersMin
                workersMax
              }
            }
            """,
            {"input": ep_input},
        )
        ep = res["saveEndpoint"]
        print(f"[deploy] Endpoint ready: {ep['id']}")
        return ep


def _upsert_env(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    replaced = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision RunPod Serverless for MiniMax-H3")
    parser.add_argument("--skip-volume", action="store_true", help="Skip creating/attaching Network Volume")
    args = parser.parse_args()

    settings = get_client_settings()
    prov = RunPodProvisioner(settings)
    acct = prov.get_account_state()
    balance = acct.get("clientBalance", 0.0)
    print(f"[deploy] RunPod account balance: ${balance:.4f}")

    tpl = prov.ensure_template(acct.get("podTemplates") or [])

    vol: dict[str, Any] | None = None
    if not args.skip_volume:
        try:
            vol = prov.ensure_network_volume(acct.get("networkVolumes") or [])
        except Exception as exc:
            print(f"[deploy] Volume creation failed (likely needs balance > $0): {exc}", file=sys.stderr)
            raise

    ep = prov.ensure_endpoint(
        existing=acct.get("endpoints") or [],
        template_id=tpl["id"],
        network_volume_id=vol["id"] if vol else None,
    )

    state = {"template": tpl, "network_volume": vol, "endpoint": ep}
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _upsert_env(Path(".env"), "RUNPOD_ENDPOINT_ID", ep["id"])
    print(f"\n[deploy] Saved state to {STATE_FILE} and RUNPOD_ENDPOINT_ID={ep['id']} to .env")
    print(f"[deploy] Next step: run `uv run minimax-h3 warmup` to sync ~37.3 GB weights onto the volume once.")


if __name__ == "__main__":
    main()

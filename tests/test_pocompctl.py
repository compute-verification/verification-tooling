from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys


MODULE_PATH = pathlib.Path(__file__).parents[1] / "ops" / "pocompctl.py"
SPEC = importlib.util.spec_from_file_location("pocompctl", MODULE_PATH)
assert SPEC and SPEC.loader
pocompctl = importlib.util.module_from_spec(SPEC)
sys.modules["pocompctl"] = pocompctl
SPEC.loader.exec_module(pocompctl)


class FakeVast:
    def __init__(self) -> None:
        self.destroyed: list[str] = []
        self.created = 0

    def choose_offer(self, _query: str) -> str:
        return "offer-1"

    def create(
        self, _offer: str, image: str, _disk: int, _public_key: str
    ) -> str:
        assert "@sha256:" in image
        self.created += 1
        return str(100 + self.created)

    def destroy(self, instance_id: str) -> dict[str, bool]:
        self.destroyed.append(instance_id)
        return {"success": True}

    def wait_running(
        self, instance_id: str, _timeout_seconds: int, _poll_seconds: int
    ) -> dict[str, str]:
        return {"id": instance_id, "actual_status": "running"}

    def wait_destroyed(
        self, instance_id: str, _timeout_seconds: int, _poll_seconds: int
    ) -> None:
        assert instance_id in self.destroyed


def args(tmp_path: pathlib.Path) -> argparse.Namespace:
    key = tmp_path / "vast.pub"
    key.write_text("ssh-ed25519 AAAAC3Nza test\n")
    return argparse.Namespace(
        pod_id="pod-1",
        image="registry/pocomp@sha256:" + "a" * 64,
        ssh_public_key=str(key),
        disk_gb=80,
        offer_query="gpu_name=H100_SXM",
        boot_timeout_seconds=1,
        destroy_timeout_seconds=1,
        poll_seconds=0,
        state=tmp_path / "pod.json",
        certificate=tmp_path / "erasure.json",
    )


def test_rotate_destroys_before_replacement_and_emits_experimental_certificate(
    tmp_path: pathlib.Path,
) -> None:
    fake = FakeVast()
    options = args(tmp_path)
    old = pocompctl.provision(options, fake)
    certificate = pocompctl.rotate(options, fake)

    assert fake.destroyed == [old.instance_id]
    body = certificate["certificate"]
    assert body["kind"] == "VastDestroyReplace"
    assert body["old_incarnation_id"] != body["new_incarnation_id"]
    assert body["old_destroyed_at_ns"] <= body["new_started_at_ns"]
    assert certificate["signature"] == []


def test_unpinned_image_is_rejected(tmp_path: pathlib.Path) -> None:
    fake = FakeVast()
    options = args(tmp_path)
    options.image = "registry/pocomp:latest"
    try:
        pocompctl.provision(options, fake)
    except pocompctl.LifecycleError:
        pass
    else:
        raise AssertionError("mutable image tag was accepted")

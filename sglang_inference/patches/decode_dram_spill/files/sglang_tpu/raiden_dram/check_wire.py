# SPDX-License-Identifier: Apache-2.0
"""Check the private wire subset against a local tpu-sync source checkout.

Requires grpcio-tools for this developer check only, not for serving.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2

from . import wire_pb2
from .client import REGISTRY_SERVICE, STORE_SERVICE


def check(source: Path) -> int:
    protos = [
        "tpu_sync/kv_cache/global_registry/global_registry.proto",
        "tpu_sync/proto/kv_cache_store_service.proto",
    ]
    with tempfile.TemporaryDirectory(prefix="sglang-raiden-wire-") as directory:
        output = Path(directory) / "upstream.pb"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{source}",
                f"--descriptor_set_out={output}",
                "--include_imports",
                *protos,
            ],
            cwd=source,
            check=True,
        )
        upstream = descriptor_pb2.FileDescriptorSet.FromString(output.read_bytes())
    local = descriptor_pb2.FileDescriptorProto.FromString(
        wire_pb2.DESCRIPTOR.serialized_pb
    )
    messages, enums, services = {}, {}, {}

    def collect(container, prefix=""):
        for message in getattr(
            container, "message_type", getattr(container, "nested_type", [])
        ):
            name = prefix + message.name
            if name in messages:
                raise ValueError(f"Ambiguous upstream message {name}")
            messages[name] = message
            collect(message, name + ".")
        for enum in container.enum_type:
            enums[prefix + enum.name] = enum

    for file in upstream.file:
        collect(file)
        for service in file.service:
            services[f"{file.package}.{service.name}"] = service

    checked = 0

    def compare(container, prefix=""):
        nonlocal checked
        for message in getattr(
            container, "message_type", getattr(container, "nested_type", [])
        ):
            name = prefix + message.name
            original = messages[name]
            fields = {field.number: field for field in original.field}
            for field in message.field:
                other = fields[field.number]
                for attribute in ("name", "type", "label", "proto3_optional"):
                    if getattr(field, attribute) != getattr(other, attribute):
                        raise ValueError(f"{name}.{field.name}: {attribute} differs")
                # Package names intentionally differ; leaf type identities and
                # every included message's fields are checked independently.
                if (
                    field.type_name.rsplit(".", 1)[-1]
                    != other.type_name.rsplit(".", 1)[-1]
                ):
                    raise ValueError(f"{name}.{field.name}: referenced type differs")
                for candidate, owner in ((field, message), (other, original)):
                    if candidate.HasField("oneof_index"):
                        oneof = owner.oneof_decl[candidate.oneof_index].name
                    else:
                        oneof = None
                    if candidate is field:
                        expected_oneof = oneof
                    elif oneof != expected_oneof:
                        raise ValueError(f"{name}.{field.name}: oneof differs")
                checked += 1
            compare(message, name + ".")
        for enum in container.enum_type:
            name = prefix + enum.name
            if [(x.name, x.number) for x in enum.value] != [
                (x.name, x.number) for x in enums[name].value
            ]:
                raise ValueError(f"{name}: enum values differ")

    compare(local)
    methods = {
        REGISTRY_SERVICE: ["Lookup", "ResolveStore", "RegisterKVTransferSpec"],
        STORE_SERVICE: ["Fetch", "WriteRemote", "PollWriteRemote"],
    }
    for service_name, names in methods.items():
        actual = {method.name: method for method in services[service_name].method}
        for name in names:
            method = actual[name]
            response = (
                "WriteRemoteEvent" if name == "WriteRemote" else name + "Response"
            )
            if (
                method.input_type.rsplit(".", 1)[-1] != name + "Request"
                or method.output_type.rsplit(".", 1)[-1] != response
                or method.client_streaming
                or method.server_streaming != (name == "WriteRemote")
            ):
                raise ValueError(f"{service_name}/{name}: RPC signature differs")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpu-sync-source", required=True, type=Path)
    args = parser.parse_args()
    count = check(args.tpu_sync_source.resolve())
    print(f"Compatible: {count} fields, enum values, and all 6 RPC signatures")


if __name__ == "__main__":
    main()

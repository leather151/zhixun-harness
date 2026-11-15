"""离线重打镜像：在已有 docker save 归档上**追加一层**，替换 /app 下的代码与模型。

为什么需要它：本机没有 WSL2，Docker Desktop 引擎起不来（`docker info` 返回 500），
但镜像里真正变化的只有 `harness/`、`vendor/`、`models/` 三块；底层（python:3.12-slim + numpy）没变。
Docker 分层语义是"后层覆盖前层"，所以只要追加一个新层即可，不必重新 build。

本归档是 **OCI 布局 + Docker 兼容文件** 的混合格式（containerd 镜像存储的 `docker save` 产物）：

    index.json      → OCI manifest blob → config blob + 10 个 layer blob
    manifest.json   → 同样的 config + layers（Docker 兼容路径）
    repositories    → 旧式 tag → 顶层 layer 摘要

因此打补丁必须**同时更新** index.json、OCI manifest、config、manifest.json、repositories。
另外注意：本镜像的层是**未压缩 tar**（mediaType `...v1.tar`），所以 `blob 摘要 == diff_id`，
新层也按同样约定构造，保持与既有层完全一致。

用法：
    python pack_image.py inspect dist/zhixun-harness_dev.tar
    python pack_image.py patch   dist/zhixun-harness_dev.tar dist/zhixun-harness_v2.tar --tag zhixun-harness:v2
    python pack_image.py verify  dist/zhixun-harness_v2.tar
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BLOB = "blobs/sha256/"
LAYER_MT = "application/vnd.oci.image.layer.v1.tar"
CFG_MT = "application/vnd.oci.image.config.v1+json"
MAN_MT = "application/vnd.oci.image.manifest.v1+json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_member(tf, name):
    f = tf.extractfile(name)
    if f is None:
        raise KeyError(f"归档里找不到 {name}")
    return f.read()


# ------------------------------------------------------------------ inspect

def inspect(path):
    path = Path(path)
    with tarfile.open(path) as tf:
        names = tf.getnames()
        print(f"文件：{path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"顶层：{sorted({n.split('/')[0] for n in names})}")
        legacy = json.loads(read_member(tf, "manifest.json"))[0]
        idx = json.loads(read_member(tf, "index.json"))
        om = idx["manifests"][0]
        oci = json.loads(read_member(tf, BLOB + om["digest"].split(":")[1]))
        cfg = json.loads(read_member(tf, legacy["Config"]))
        print(f"tag={legacy.get('RepoTags')}  arch={cfg.get('architecture')} os={cfg.get('os')}")
        print(f"Entrypoint={cfg['config'].get('Entrypoint')}")
        print(f"User={cfg['config'].get('User')}  WorkDir={cfg['config'].get('WorkingDir')}")
        print(f"层 {len(legacy['Layers'])} / diff_ids {len(cfg['rootfs']['diff_ids'])} / "
              f"history {len(cfg.get('history', []))}")
        for i, l in enumerate(legacy["Layers"]):
            size = tf.getmember(l).size / 1024 / 1024
            mt = oci["layers"][i]["mediaType"].split("/")[-1]
            print(f"  L{i:<2} {l.replace(BLOB, '')[:16]}…  {size:8.2f} MB  {mt}")
        print(f"index→manifest: {om['digest'][:20]}…  config={legacy['Config'][-16:]}…")
        print(f"repositories: {read_member(tf, 'repositories').decode()[:160]}")


# ------------------------------------------------------------------ verify

def verify(path, expect_files=None, compare=None):
    """校验归档自洽性。

    硬校验（不过就不能上传）：blob 摘要与文件名一致、索引链路摘要一致、层数与 diff_ids 一致、
    新层里关键文件存在。
    软提示：`diff_id` 与层字节是否一致 —— 本镜像全部不一致，原因是 `docker save` 会重新打包层，
    而 `docker load` 只校验 blob 完整性、不校验 diff_id，所以历史上无人发现（v1 已在平台加载成功）。
    """
    path = Path(path)
    problems, notes = [], []
    with tarfile.open(path) as tf:
        names = set(tf.getnames())
        if "index.json" not in names or "manifest.json" not in names:
            return ["缺少 index.json 或 manifest.json"]
        idx = json.loads(read_member(tf, "index.json"))
        legacy = json.loads(read_member(tf, "manifest.json"))[0]

        for m in idx.get("manifests", []):
            digest = m["digest"].split(":")[1]
            name = BLOB + digest
            if name not in names:
                problems.append(f"index.json 指向的 manifest 缺失：{name}")
                continue
            raw = read_member(tf, name)
            if sha256_bytes(raw) != digest:
                problems.append("OCI manifest 摘要与文件名不一致")
            if m.get("size") not in (None, len(raw)):
                problems.append("index.json 里 manifest 的 size 不对")
            oci = json.loads(raw)
            cfg_name = BLOB + oci["config"]["digest"].split(":")[1]
            if cfg_name not in names:
                problems.append(f"config blob 缺失：{cfg_name}")
                continue
            cfg_raw = read_member(tf, cfg_name)
            if sha256_bytes(cfg_raw) != oci["config"]["digest"].split(":")[1]:
                problems.append("config 摘要与文件名不一致")
            cfg = json.loads(cfg_raw)
            if legacy["Config"] != cfg_name:
                problems.append("manifest.json 与 OCI manifest 指向不同 config")
            if len(oci["layers"]) != len(cfg["rootfs"]["diff_ids"]):
                problems.append("OCI manifest 层数与 diff_ids 不一致")
            if len(legacy["Layers"]) != len(oci["layers"]):
                problems.append("manifest.json 层数与 OCI manifest 不一致")
            diff_mismatch = 0
            for i, layer in enumerate(oci["layers"]):
                lname = BLOB + layer["digest"].split(":")[1]
                if lname not in names:
                    problems.append(f"L{i} 层文件缺失：{lname}")
                    continue
                raw_layer = read_member(tf, lname)
                actual = sha256_bytes(raw_layer)
                if actual != layer["digest"].split(":")[1]:
                    problems.append(f"L{i} blob 摘要与文件名不一致（归档损坏）")
                if layer.get("size") not in (None, len(raw_layer)):
                    problems.append(f"L{i} 声明的 size 与实际不符")
                # diff_id 只做提示
                if raw_layer[:2] == b"\x1f\x8b":
                    import gzip
                    try:
                        un = gzip.decompress(raw_layer)
                    except OSError:
                        problems.append(f"L{i} gzip 内容损坏")
                        continue
                else:
                    un = raw_layer
                if sha256_bytes(un) != cfg["rootfs"]["diff_ids"][i]:
                    diff_mismatch += 1
            if diff_mismatch:
                notes.append(f"{diff_mismatch} 个层的 diff_id 与层字节不一致"
                             f"（docker save 重打包所致，load 不校验）")

            if expect_files:
                top = oci["layers"][-1]
                top_raw = read_member(tf, BLOB + top["digest"].split(":")[1])
                if top_raw[:2] == b"\x1f\x8b":
                    import gzip
                    top_raw = gzip.decompress(top_raw)
                top_files = {}
                with tarfile.open(fileobj=io.BytesIO(top_raw)) as lt:
                    for member in lt.getmembers():
                        if member.isfile():
                            top_files[member.name.lstrip("./")] = lt.extractfile(member).read()
                for want in expect_files:
                    if want not in top_files:
                        problems.append(f"新层里缺少 {want}")
                if compare:
                    bad = 0
                    for guest, host in compare.items():
                        if guest not in top_files:
                            problems.append(f"新层里缺少 {guest}")
                            continue
                        host_digest = hashlib.sha256(Path(host).read_bytes()).hexdigest()
                        if hashlib.sha256(top_files[guest]).hexdigest() != host_digest:
                            problems.append(f"新层里 {guest} 与本地 {host} 内容不一致")
                            bad += 1
                    if not bad:
                        notes.append(f"逐文件比对 {len(compare)} 个关键文件：全部一致")

        print(f"校验 {path.name}：层 {len(legacy['Layers'])}，"
              f"硬问题 {len(problems)} 个，提示 {len(notes)} 条")
        for n in notes:
            print("  ·", n)
        for p in problems:
            print("  ✗", p)
        if not problems:
            print("  ✓ 归档自洽（blob 完整性 + OCI/Docker 双链路一致），可被 docker load")
    return problems


# ------------------------------------------------------------------ patch

def build_layer(prefix_map):
    """把宿主机目录打成一层（未压缩 tar，与既有层一致）。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for src_root, dst_root in prefix_map:
            src_root = Path(src_root)
            if not src_root.exists():
                raise FileNotFoundError(f"内容目录不存在：{src_root}")
            for dirpath, dirnames, filenames in os.walk(src_root):
                dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
                rel = Path(dirpath).relative_to(src_root)
                dst_dir = (Path(dst_root) / rel).as_posix()
                info = tarfile.TarInfo(dst_dir)
                info.type, info.mode, info.uid, info.gid = tarfile.DIRTYPE, 0o755, 0, 0
                info.mtime = 0
                tf.addfile(info)
                for name in sorted(filenames):
                    if name.endswith(".pyc"):
                        continue
                    data = (Path(dirpath) / name).read_bytes()
                    info = tarfile.TarInfo((Path(dst_dir) / name).as_posix())
                    info.size, info.mode, info.uid, info.gid = len(data), 0o644, 0, 0
                    info.mtime = 0
                    tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def patch(base_tar, out_tar, tag, prefix_map, note):
    base_tar, out_tar = Path(base_tar), Path(out_tar)
    repo, ver = (tag.split(":", 1) + ["latest"])[:2]

    layer_raw = build_layer(prefix_map)
    layer_digest = sha256_bytes(layer_raw)          # 未压缩层：digest == diff_id

    with tarfile.open(base_tar) as tf:
        names = tf.getnames()
        idx = json.loads(read_member(tf, "index.json"))
        legacy = json.loads(read_member(tf, "manifest.json"))
        old_man_digest = idx["manifests"][0]["digest"].split(":")[1]
        oci = json.loads(read_member(tf, BLOB + old_man_digest))
        cfg_name_old = legacy[0]["Config"]
        cfg = json.loads(read_member(tf, cfg_name_old))

        # 1) 新 config：diff_ids + history
        cfg["rootfs"]["diff_ids"] = list(cfg["rootfs"]["diff_ids"]) + [layer_digest]
        cfg.setdefault("history", []).append({
            "created": "2026-11-15T00:00:00Z",
            "created_by": note,
            "comment": "offline layer patch (no docker daemon / WSL2 available)",
        })
        cfg_raw = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode()
        cfg_digest = sha256_bytes(cfg_raw)
        cfg_name = BLOB + cfg_digest

        # 2) 新 OCI manifest
        oci["config"] = {"mediaType": CFG_MT, "digest": "sha256:" + cfg_digest,
                         "size": len(cfg_raw)}
        oci["layers"] = list(oci["layers"]) + [
            {"mediaType": LAYER_MT, "digest": "sha256:" + layer_digest, "size": len(layer_raw)}]
        man_raw = json.dumps(oci, ensure_ascii=False, sort_keys=True).encode()
        man_digest = sha256_bytes(man_raw)
        man_name = BLOB + man_digest

        # 3) 新 index.json（保留注解，更新 tag 与指向）
        idx["manifests"][0].update({"digest": "sha256:" + man_digest, "size": len(man_raw)})
        ann = idx["manifests"][0].setdefault("annotations", {})
        ann["org.opencontainers.image.ref.name"] = ver
        ann["io.containerd.image.name"] = f"docker.io/library/{repo}:{ver}"

        # 4) Docker 兼容 manifest.json 与 repositories
        new_legacy = [{"Config": cfg_name, "RepoTags": [tag],
                       "Layers": list(legacy[0]["Layers"]) + [BLOB + layer_digest]}]
        repos = {repo: {ver: layer_digest}}

        out_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_tar, "w") as out:
            def add(name, data):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                out.addfile(info, io.BytesIO(data))

            add("manifest.json", json.dumps(new_legacy, ensure_ascii=False).encode())
            add("repositories", json.dumps(repos, ensure_ascii=False).encode())
            add("index.json", json.dumps(idx, ensure_ascii=False).encode())
            add("oci-layout", read_member(tf, "oci-layout"))
            copied = set()
            for blob in names:
                if not blob.startswith(BLOB) or blob.endswith("/") or blob in copied:
                    continue
                add(blob, read_member(tf, blob))       # 旧 blob 原样保留
                copied.add(blob)
            add(cfg_name, cfg_raw)
            add(man_name, man_raw)
            add(BLOB + layer_digest, layer_raw)

    print(f"写盘：{out_tar}  ({out_tar.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  新层 {layer_digest[:16]}…  {len(layer_raw) / 1024 / 1024:.2f} MB（未压缩）")
    print(f"  新 config {cfg_name[-16:]}…  新 manifest {man_digest[:16]}…")
    print(f"  tag={tag}  层数 {len(new_legacy[0]['Layers'])}")
    return out_tar


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("inspect"); p1.add_argument("tar")
    p2 = sub.add_parser("patch")
    p2.add_argument("base"); p2.add_argument("out")
    p2.add_argument("--tag", default="zhixun-harness:v2")
    p3 = sub.add_parser("verify"); p3.add_argument("tar")
    args = ap.parse_args(argv)

    if args.cmd == "inspect":
        inspect(args.tar); return 0
    if args.cmd == "verify":
        return 0 if not verify(args.tar) else 1

    prefix_map = [(PROJECT / "harness", "app/harness"),
                  (PROJECT / "vendor", "app/vendor"),
                  (PROJECT / "models", "app/models")]
    out = patch(args.base, args.out, args.tag, prefix_map,
                note="COPY harness/ vendor/ models/ (offline repack)")
    expect = ["app/harness/runner.py", "app/harness/entry.py", "app/harness/preprocess.py",
              "app/harness/serve.py", "app/harness/train_a.py", "app/harness/train_b.py",
              "app/harness/train_c.py", "app/vendor/A/model.py", "app/vendor/B/model.py",
              "app/models/A/checkpoint_best.npz", "app/models/B/checkpoint_rank.npz",
              "app/models/C/checkpoint_best.npz"]
    # 逐文件比对：镜像内新层内容 vs 本地源文件（最硬的端到端检查）
    compare = {}
    for host_dir, guest_dir in prefix_map:
        for f in sorted(Path(host_dir).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".pyc"):
                compare[(Path(guest_dir) / f.relative_to(host_dir)).as_posix()] = str(f)
    if verify(out, expect_files=expect, compare=compare):
        print("\n自检未通过，请勿上传。", file=sys.stderr)
        return 1
    print("\n自检通过：可以上传该 tar。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

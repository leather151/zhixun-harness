#!/usr/bin/env bash
# ============================================================================
#  在 Ubuntu 虚拟机里：装 docker → 配镜像加速 → 拉基础镜像 → build → 自检 → save
#  由宿主机 vm-build.ps1 调用：  bash /tmp/zhixun/vm-build.sh /tmp/zhixun zhixun-harness:dev
#
#  为什么全部逻辑放这里：PowerShell 传参给 ssh 会吃掉双引号，远端 bash 直接语法错。
#  所以宿主机只负责"把文件传进来 + 执行这个脚本"，所有复杂逻辑都在这个文件里。
# ============================================================================
set -uo pipefail

CTX="${1:-/tmp/zhixun}"
TAG="${2:-zhixun-harness:dev}"
MIRROR="${3:-https://docker.1panel.live}"
PIP_INDEX="${4:-https://pypi.tuna.tsinghua.edu.cn/simple}"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if sudo -n true 2>/dev/null; then SUDO="sudo"; else
    echo "[FATAL] 当前用户 $(id -un) 没有免密 sudo" >&2
    exit 3
  fi
fi

step() { echo ""; echo "=== $* ==="; }
ok()   { echo "  [OK] $*"; }
die()  { echo "  [FAIL] $*" >&2; exit 1; }

step "[vm 1/7] 环境"
. /etc/os-release 2>/dev/null || true
echo "  OS=${PRETTY_NAME:-unknown}  arch=$(uname -m)  磁盘剩余=$(df -h / | awk 'NR==2{print $4}')"
[ -f "$CTX/Dockerfile" ] || die "构建上下文不完整：$CTX 里没有 Dockerfile"

step "[vm 2/7] 安装 / 检查 docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "  未检测到 docker，安装 docker.io（走 /etc/apt 里的国内源）"
  # 等 dpkg 锁（刚开机常有 unattended-upgrades 在跑）
  for _ in $(seq 1 24); do
    if $SUDO fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then sleep 5; else break; fi
  done
  $SUDO systemctl stop unattended-upgrades >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq docker.io > /tmp/zhixun-apt.log 2>&1
  if [ $? -ne 0 ]; then
    tail -15 /tmp/zhixun-apt.log
    die "apt 安装 docker.io 失败"
  fi
fi
ok "$(docker --version)"

step "[vm 3/7] 配置镜像加速 + 启动 daemon"
$SUDO mkdir -p /etc/docker
# containerd-snapshotter=false：让 docker save 产出**传统格式**（manifest.json + repositories + 每层目录），
# 这种 tar 在 Docker 20/24/25/29 上都能 load；Docker 29 默认的新 OCI 格式老平台可能不认。
cat > /tmp/zhixun-daemon.json <<JSON
{
  "registry-mirrors": ["$MIRROR"],
  "features": { "containerd-snapshotter": false }
}
JSON
$SUDO cp /tmp/zhixun-daemon.json /etc/docker/daemon.json
echo "  daemon.json -> registry-mirrors: $MIRROR ; containerd-snapshotter: false"

# 先试 systemd；不行就手动后台起 dockerd（本机实测：systemd 那套会挂，手动没问题）
$SUDO systemctl daemon-reload >/dev/null 2>&1 || true
$SUDO systemctl restart docker >/dev/null 2>&1 || true
sleep 4
if ! $SUDO docker info >/dev/null 2>&1; then
  echo "  systemd 启动失败，改为手动后台启动 dockerd"
  $SUDO pkill -x dockerd >/dev/null 2>&1 || true
  sleep 1
  $SUDO sh -c 'nohup dockerd > /tmp/dockerd.log 2>&1 &'
  sleep 8
  if ! $SUDO docker info >/dev/null 2>&1; then
    tail -20 /tmp/dockerd.log 2>/dev/null
    die "docker daemon 起不来"
  fi
fi
ok "$($SUDO docker info --format 'ServerVersion={{.ServerVersion}} CgroupV{{.CgroupVersion}} Driver={{.Driver}}')"

step "[vm 4/7] 拉基础镜像 python:3.12-slim"
if $SUDO docker image inspect python:3.12-slim >/dev/null 2>&1; then
  ok "已存在，跳过拉取"
else
  if ! timeout 600 $SUDO docker pull python:3.12-slim; then
    die "拉基础镜像失败（镜像源 $MIRROR 可能失效，换一个再试：bash $0 $CTX $TAG https://其他镜像)"
  fi
  ok "拉取完成"
fi

step "[vm 5/7] 构建镜像 $TAG"
cd "$CTX" || die "进不去 $CTX"
if ! $SUDO docker build --build-arg "PIP_INDEX=$PIP_INDEX" -t "$TAG" .; then
  die "docker build 失败"
fi
$SUDO docker image inspect "$TAG" --format '  镜像Id={{.Id}}  大小={{.Size}} bytes'
ok "构建完成"

step "[vm 6/7] 容器内自检（三包各 1 条真实样本推理）"
$SUDO docker run --rm -e HARNESS_MODE=selftest "$TAG" || die "容器内 selftest 失败"
ok "自检通过"

step "[vm 7/7] 导出 docker save 归档"
mkdir -p dist
OUT="dist/$(echo "$TAG" | tr ':/' '__').tar"
$SUDO rm -f "$OUT"
$SUDO docker save "$TAG" -o "$OUT" || die "docker save 失败"
$SUDO chown "$(id -u):$(id -g)" "$OUT"
# 校验归档格式：传统格式会有 repositories 文件
if tar -tf "$OUT" 2>/dev/null | grep -qx "repositories"; then
  ok "格式=传统（含 repositories，兼容 Docker 20+）"
else
  echo "  [WARN] 未检测到 repositories，可能是新 OCI 格式，老平台 docker load 可能不认"
fi
ok "已导出 $CTX/$OUT ($(du -h "$OUT" | cut -f1))"
echo ""
echo "VMBUILD-DONE $CTX/$OUT"

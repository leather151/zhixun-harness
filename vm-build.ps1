# ============================================================================
#  备用构建路径：在已有的 Ubuntu 虚拟机里构建 Harness 镜像
#
#    .\vm-build.ps1                     # 自动找 IP → 传工程 → VM 内构建 → 取回 tar
#    .\vm-build.ps1 -Ip 192.168.30.133  # 直接指定 IP（vmware-tools 没装时用它）
#    .\vm-build.ps1 -KeepRunning        # 结束后不关虚拟机
#
#  为什么要有这条路径：宿主机 Docker 需要 WSL2（要管理员 + 重启）。这条路径都不需要。
#  已在 Ubuntu24.04 VM 上实测：VM 用 NAT(192.168.30.x)，DHCP 租约 hostname=ros2-vm。
# ============================================================================
param(
    [string]$Vmx = "D:\VM\Ubuntu24.04\Ubuntu24.04-ROS2.vmx",
    [string]$SshUser = "zwb",
    [string]$KeyPath = "D:\VM\Ubuntu24.04\id_ed25519",
    [string]$Tag = "zhixun-harness:dev",
    [string]$Ip = "",
    [string]$Mirror = "https://docker.1panel.live",
    [string]$PipIndex = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [switch]$KeepRunning,
    [int]$WaitSeconds = 180
)
Set-Location -Path $PSScriptRoot
function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Ok($t)   { Write-Host "  [OK] $t" -ForegroundColor Green }
function Bad($t)  { Write-Host "  [FAIL] $t" -ForegroundColor Red }

$vmrun = "C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"
foreach ($p in @($vmrun, $Vmx, $KeyPath)) { if (-not (Test-Path $p)) { throw "找不到：$p" } }

# 统一的 ssh/scp 参数（一定要带 BatchMode，避免卡在交互提示上）
$sshOpt = @("-i", $KeyPath, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=8", "-o", "BatchMode=yes")

function Invoke-Native {
    param([string]$Exe, [string[]]$CmdArgs)
    # 关键：把 stderr 合进来并吞掉 PowerShell 的"原生命令错误"，只返回内容
    $out = & $Exe @CmdArgs 2>&1 | ForEach-Object { $_.ToString() }
    return @{ code = $LASTEXITCODE; out = $out }
}

# ---------------------------------------------------------------- 1. 启动 VM
Step "[1/6] 启动虚拟机"
$list = (& $vmrun list 2>&1 | Out-String)
if ($list -match [regex]::Escape((Split-Path $Vmx -Leaf))) { Ok "已在运行" }
else {
    & $vmrun start $Vmx nogui 2>&1 | Out-Null
    Ok "已发出启动命令（等网络就绪）"
    Start-Sleep -Seconds 10
}

# ---------------------------------------------------------------- 2. 找 IP
Step "[2/6] 定位虚拟机 IP"
if (-not $Ip) {
    $r = Invoke-Native $vmrun @("getGuestIPAddress", $Vmx)
    if ($r.code -eq 0) {
        $cand = ($r.out | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1)
        if ($cand) { $Ip = $cand.Trim(); Ok "vmware-tools 上报：$Ip" }
    }
    if (-not $Ip) {
        # vmware-tools 没装时的兜底：从 DHCP 租约里按 VM 的 MAC 找地址
        $mac = (Select-String -Path $Vmx -Pattern 'ethernet0\.generatedAddress\s*=\s*"([0-9a-fA-F:]{17})"').Matches
        $leases = "C:\ProgramData\VMware\vmnetdhcp.leases"
        if ($mac.Count -gt 0 -and (Test-Path $leases)) {
            $macAddr = $mac[0].Groups[1].Value.ToLower()
            $blocks = (Get-Content $leases -Raw) -split "lease "
            $hit = $blocks | Where-Object { $_ -match [regex]::Escape($macAddr) } | Select-Object -First 1
            if ($hit -and $hit -match '^\s*(\d+\.\d+\.\d+\.\d+)') { $Ip = $Matches[1] }
            if ($Ip) { Ok "DHCP 租约兜底（MAC $macAddr）：$Ip" }
        }
    }
}
if (-not $Ip) {
    Bad "拿不到 IP"
    Write-Host "  在 VM 窗口里执行 ip -4 addr show 看地址，然后用 .\vm-build.ps1 -Ip <地址>" -ForegroundColor Yellow
    throw "无法定位虚拟机 IP"
}

# ---------------------------------------------------------------- 3. 等 SSH
Step "[3/6] 等待 SSH"
$deadline = (Get-Date).AddSeconds($WaitSeconds); $sshOk = $false
while ((Get-Date) -lt $deadline) {
    $r = Invoke-Native "ssh" ($sshOpt + @("$SshUser@$Ip", "true"))
    if ($r.code -eq 0) { $sshOk = $true; break }
    Start-Sleep -Seconds 4; Write-Host "  ...等待 sshd" -ForegroundColor DarkGray
}
if (-not $sshOk) { throw "SSH 连不上 $SshUser@$Ip（确认 VM 里 sshd 在跑、密钥已授权）" }
Ok "SSH 就绪 ($SshUser@$Ip)"

# ---------------------------------------------------------------- 4. 打包上传
Step "[4/6] 打包构建上下文并上传"
$stage = Join-Path $env:TEMP "zhixun-stage"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "data") | Out-Null
foreach ($item in "harness", "vendor", "models", "Dockerfile", ".dockerignore", "vm-build.sh") {
    Copy-Item (Join-Path $PSScriptRoot $item) (Join-Path $stage $item) -Recurse -Force
}
Copy-Item (Join-Path $PSScriptRoot "data\public") (Join-Path $stage "data\public") -Recurse -Force
$tarLocal = Join-Path $env:TEMP "zhixun-ctx.tar"
Remove-Item $tarLocal -Force -ErrorAction SilentlyContinue
& tar -cf $tarLocal -C $stage .
if ($LASTEXITCODE -ne 0) { throw "打包失败" }
Ok ("上下文 " + [math]::Round((Get-Item $tarLocal).Length / 1MB, 1) + " MB")

$r = Invoke-Native "scp" ($sshOpt + @($tarLocal, "${SshUser}@${Ip}:/tmp/zhixun-ctx.tar"))
if ($r.code -ne 0) { $r.out | Select-Object -Last 5; throw "scp 上传失败" }
$r = Invoke-Native "ssh" ($sshOpt + @("$SshUser@$Ip", "rm -rf /tmp/zhixun && mkdir -p /tmp/zhixun && tar -xf /tmp/zhixun-ctx.tar -C /tmp/zhixun && ls /tmp/zhixun"))
if ($r.code -ne 0) { $r.out | Select-Object -Last 5; throw "解包失败" }
Ok "已解包到 VM:/tmp/zhixun"

# ---------------------------------------------------------------- 5. VM 内构建
Step "[5/6] VM 内构建（装 docker + 拉基础镜像，首次可能 5–15 分钟）"
$remote = "bash /tmp/zhixun/vm-build.sh /tmp/zhixun $Tag $Mirror $PipIndex"
$r = Invoke-Native "ssh" ($sshOpt + @("$SshUser@$Ip", $remote))
$r.out | ForEach-Object { Write-Host "  $_" }
if ($r.code -ne 0) {
    throw "VM 内构建失败（退出码 $($r.code)）。可登录 VM 手动复现：$remote"
}

# ---------------------------------------------------------------- 6. 取回
Step "[6/6] 取回镜像归档"
New-Item -ItemType Directory -Force -Path "dist" | Out-Null
$safe = ($Tag -replace '[:/]', '_')
$r = Invoke-Native "scp" ($sshOpt + @("${SshUser}@${Ip}:/tmp/zhixun/dist/$safe.tar", "dist\$safe.tar"))
if ($r.code -ne 0) { $r.out | Select-Object -Last 5; throw "scp 取回失败" }
Ok ("已取回 dist\$safe.tar (" + [math]::Round((Get-Item "dist\$safe.tar").Length / 1MB, 1) + " MB)")

if (-not $KeepRunning) {
    Write-Host "`n  关闭虚拟机 ..." -ForegroundColor DarkGray
    & $vmrun stop $Vmx soft 2>&1 | Out-Null
}

Write-Host "`n完成。" -ForegroundColor Green
Write-Host "提交：作品提交 → 选择文件 → 上传 dist\$safe.tar → 刷新状态 → 新建提交版本" -ForegroundColor Yellow
Write-Host "先花 1 次「试运行」验证端到端，正式评测只有 2 次。" -ForegroundColor Yellow

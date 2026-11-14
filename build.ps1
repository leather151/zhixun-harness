# ============================================================================
#  一键构建 Harness 镜像：预检 → 构建 → 容器内自检 → 闭环自检 → docker save
#
#    .\build.ps1 -Doctor        # 只做体检，不构建（Docker 起不来时先跑这个）
#    .\build.ps1                # 构建 zhixun-harness:dev 并导出 dist\*.tar
#    .\build.ps1 -StartEngine   # 引擎没起来时，自动拉 Docker Desktop 并等它
#    .\build.ps1 -NoExport      # 只构建 + 自检，不导出
#    .\build.ps1 -Tag v2        # 自定义 tag
#
# 说明：本机 Docker Desktop 用的是 WSL2 引擎。**没装 WSL2 时 docker info 会返回 500**，
#      这是环境问题、不是脚本问题——-Doctor 会把缺什么、怎么补直接打出来。
# ============================================================================
param(
    [string]$Tag = "zhixun-harness:dev",
    [switch]$NoExport,
    [switch]$SkipSelfTest,
    [switch]$StartEngine,
    [switch]$Doctor,
    [int]$WaitSeconds = 240
)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

# ----------------------------------------------------------------- 小工具
function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Write-Bad($text)  { Write-Host "  [缺失] $text" -ForegroundColor Red }
function Write-Hint($text) { Write-Host "         $text" -ForegroundColor DarkGray }

function Find-Docker {
    $cmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    if (Test-Path $fallback) { return $fallback }
    return $null
}

function Invoke-Docker {
    # 统一走 cmd /c 并把 stderr 合进 stdout，避免 PowerShell 把原生命令的 stderr 当异常抛出
    param([string[]]$DockerArgs, [switch]$Quiet)
    $exe = $script:dockerExe
    $joined = ($DockerArgs | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join " "
    $out = & cmd /c "`"$exe`" $joined 2>&1"
    $code = $LASTEXITCODE
    if (-not $Quiet -and $out) { $out | ForEach-Object { Write-Host "         $_" -ForegroundColor DarkGray } }
    return @{ code = $code; out = ($out | Out-String).Trim() }
}

function Test-Wsl {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        return @{ ok = $false; why = "找不到 wsl.exe" }
    }
    $ver = & wsl.exe --version 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0 -and $ver -match '\d+\.\d+') {
        return @{ ok = $true; why = (($ver -split "`n")[0]).Trim() }
    }
    $list = & wsl.exe --list --quiet 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) { return @{ ok = $true; why = "已安装（无发行版，Docker Desktop 会自建）" } }
    return @{ ok = $false; why = "未安装 Linux 子系统（WSL2）" }
}

function Test-Engine {
    $r = Invoke-Docker -DockerArgs @("info", "--format", "{{.ServerVersion}}") -Quiet
    if ($r.code -eq 0 -and $r.out -and $r.out -notmatch "ERROR") {
        return @{ ok = $true; version = $r.out }
    }
    return @{ ok = $false; error = $r.out }
}

# ----------------------------------------------------------------- 预检
$script:dockerExe = Find-Docker

if ($Doctor -or -not $script:dockerExe) {
    Write-Step "体检：Docker / WSL2 / Engine"

    if ($script:dockerExe) { Write-Ok "docker CLI：$script:dockerExe" }
    else { Write-Bad "找不到 docker CLI（未安装 Docker Desktop，或安装后没重开终端）" }

    $wsl = Test-Wsl
    if ($wsl.ok) { Write-Ok "WSL2：$($wsl.why)" }
    else        { Write-Bad "WSL2：$($wsl.why)" }

    if ($script:dockerExe) {
        $eng = Test-Engine
        if ($eng.ok) { Write-Ok "Docker 引擎：ServerVersion $($eng.version)" }
        else {
            Write-Bad "Docker 引擎无响应"
            if ($eng.error) { Write-Hint $eng.error }
        }
    }

    if (-not $wsl.ok) {
        Write-Host ""
        Write-Host "  >>> 根因：Docker Desktop 用的是 WSL2 引擎，没装 WSL2 时 /info 一定返回 500。" -ForegroundColor Yellow
        Write-Host "  >>> 修复（管理员 PowerShell，然后重启一次）：" -ForegroundColor Yellow
        Write-Host "        wsl --install --no-distribution" -ForegroundColor White
        Write-Host "      重启后打开 Docker Desktop，等托盘图标变绿，再回来跑 .\build.ps1" -ForegroundColor White
        Write-Host "  >>> 如果 wsl --install 被策略/网络挡住，用备选路径：" -ForegroundColor Yellow
        Write-Host "        .\vm-build.ps1        （在已有的 Ubuntu24.04 虚拟机里构建）" -ForegroundColor White
    }
    elseif ($script:dockerExe) {
        $eng = Test-Engine
        if (-not $eng.ok) {
            Write-Host ""
            Write-Host "  >>> WSL2 在位但引擎没起来。先试：.\build.ps1 -StartEngine" -ForegroundColor Yellow
            Write-Host "      仍不行：Docker Desktop → Troubleshoot → Restart / Reset to factory defaults" -ForegroundColor White
        }
    }

    if ($Doctor) { Write-Host "`n（-Doctor 只体检，不构建）"; exit 0 }
    if (-not $script:dockerExe) { throw "Docker 不可用，见上面的修复步骤" }
}

# ----------------------------------------------------------------- 等引擎
$engine = Test-Engine
if (-not $engine.ok -and $StartEngine) {
    Write-Step "拉起 Docker Desktop 并等待引擎"
    $exe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $exe) { Start-Process $exe; Write-Host "  已启动，最多等 $WaitSeconds 秒 ..." }
    else { throw "找不到 Docker Desktop.exe" }
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        $engine = Test-Engine
        if ($engine.ok) { break }
        Write-Host "  ...仍在启动" -ForegroundColor DarkGray
    }
}
if (-not $engine.ok) {
    Write-Bad "Docker 引擎不可用"
    if ($engine.error) { Write-Hint $engine.error }
    Write-Hint "依次试：.\build.ps1 -Doctor   →   .\build.ps1 -StartEngine   →   .\vm-build.ps1"
    throw "Docker 引擎没起来，构建中止（没有产生任何镜像）"
}
Write-Ok "Docker 引擎 ServerVersion $($engine.version)"

# ----------------------------------------------------------------- 构建
Write-Step "[1/4] 构建镜像 $Tag"
$r = Invoke-Docker -DockerArgs @("build", "-t", $Tag, ".")
if ($r.code -ne 0) { throw "docker build 失败（退出码 $($r.code)）" }
Write-Ok "构建完成"

if (-not $SkipSelfTest) {
    Write-Step "[2/4] 容器内自检（代码 import + 三包各 1 条真实样本推理）"
    $r = Invoke-Docker -DockerArgs @("run", "--rm", "-e", "HARNESS_MODE=selftest", $Tag)
    if ($r.code -ne 0) { throw "容器内 selftest 失败（退出码 $($r.code)）" }

    Write-Step "[3/4] 离线闭环自检（镜像内公开数据，B 包 2 窗口，不落宿主机）"
    $r = Invoke-Docker -DockerArgs @("run", "--rm", $Tag, "--task", "B", "--stream", "official",
                                     "--windows", "2", "--swan", "off", "--out", "/tmp/selftest")
    if ($r.code -ne 0) { throw "容器内闭环自检失败（退出码 $($r.code)）" }
    Write-Ok "闭环自检通过（最终模型在镜像内 /tmp/selftest/model/B/checkpoint.npz）"
} else {
    Write-Step "[2/4][3/4] 已跳过自检"
}

# ----------------------------------------------------------------- 导出
if (-not $NoExport) {
    Write-Step "[4/4] 导出镜像归档（提交页要的就是这个 tar）"
    New-Item -ItemType Directory -Force -Path "dist" | Out-Null
    $safe = ($Tag -replace '[:/]', '_')
    $tar = Join-Path "dist" "$safe.tar"
    if (Test-Path $tar) { Remove-Item $tar -Force }
    $r = Invoke-Docker -DockerArgs @("save", $Tag, "-o", $tar)
    if ($r.code -ne 0) { throw "docker save 失败（退出码 $($r.code)）" }
    $mb = [math]::Round((Get-Item $tar).Length / 1MB, 1)
    Write-Ok "已导出：$tar  ($mb MB)"
    Write-Host "`n提交步骤：作品提交 → 选择文件 → 上传该 tar → 刷新状态 → 新建提交版本" -ForegroundColor Yellow
    Write-Host "先用 1 次「试运行」验证端到端，别直接消耗正式评测的 2 次。" -ForegroundColor Yellow
} else {
    Write-Step "[4/4] 已跳过导出"
}

Write-Host "`n完成。" -ForegroundColor Green
Write-Host "本地运行示例："
Write-Host "  docker run --rm -v `"${PWD}\data\public:/data:ro`" -v `"${PWD}\runs:/output`" $Tag --task all --stream auto --windows 4 --swan off --out /output"

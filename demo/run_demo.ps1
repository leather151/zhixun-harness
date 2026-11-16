<#
  演示视频一键执行脚本（目标总时长 < 4 分钟，留出解说余量）

  用法（在工程根目录）：
      .\demo\run_demo.ps1              # 完整演示（约 90–120 秒）
      .\demo\run_demo.ps1 -Task A      # 只演示指定任务
      .\demo\run_demo.ps1 -SkipLoop    # 跳过闭环，只看自检+协议+产物

  设计原则：
  1) 每一节都有明显的标题块，录屏时观众能看清"现在在演示什么"；
  2) 全部用本机真实数据跑，不预录、不造数字；
  3) 结尾会打印本次演示产生的产物路径与指标，方便对着讲。
#>
param(
    [string]$Task = "B",
    [int]$Windows = 4,
    [int]$Rounds = 3,
    [switch]$SkipLoop
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$demoOut = Join-Path $env:TEMP "zhixun-demo-$stamp"
New-Item -ItemType Directory -Force -Path $demoOut | Out-Null

function Section($text) {
    Write-Host ""
    Write-Host ("=" * 74) -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 74) -ForegroundColor Cyan
}
function Pause-Brief($ms = 700) { Start-Sleep -Milliseconds $ms }

# ---------------------------------------------------------------- 0. 片头
Section "0 · 智训引擎 · 工业模型自进化系统（演示开始）"
Write-Host "  数据：UCI Gas / XJTU-SY 轴承 / NASA C-MAPSS（官方公开部分）"
Write-Host "  闭环：漂移监测 -> 诊断归因 -> 实验优化 -> 评估上线 -> 回滚"
Write-Host "  策略：PyTRiO 微调决策模型 + 规则兜底；模型层：CPU 基线自迭代"
Pause-Brief 1200

# ---------------------------------------------------------------- 1. 自检
Section "1 · 镜像自检：三个任务的代码 / 模型 / 数据是否可达"
$env:HARNESS_MODE = "selftest"
python -u harness/entry.py
Pause-Brief 800

# ---------------------------------------------------------------- 2. 闭环
if (-not $SkipLoop) {
    Section "2 · 自进化闭环实跑：任务 $Task，$Windows 个数据流窗口"
    Write-Host "  每一步都会打印：当窗指标 / 参照指标 / 漂移强度 / 决策动作 / 是否上线"
    $env:HARNESS_MODE = "loop"
    python -u harness/entry.py --task $Task --stream train --windows $Windows `
        --policy auto --swan off --out (Join-Path $demoOut "runs")
    Pause-Brief 800
}

# ---------------------------------------------------------------- 3. 协议模拟
Section "3 · 平台多轮协议本地复刻：候选 -> 评估 -> 采用 / 保留"
Write-Host "  说明：本地只有公开验证集，隐藏测试流不可得，所以这是行为复刻而非成绩"
python -u demo/round_protocol_sim.py --task A --rounds $Rounds --windows 2
Pause-Brief 800

# ---------------------------------------------------------------- 4. 预测协议
Section "4 · 官方 predict 协议 + 交付产物审计"
$rows = @()
$base = "D:\人工智能大赛\02_数据"
foreach ($pair in @(@("A_气体传感器", "A"), @("B_滚动轴承RUL", "B"), @("C_涡扇发动机RUL", "C"))) {
    $p = Join-Path $base "$($pair[0])\validation.jsonl"
    if (Test-Path $p) { $rows += (Get-Content $p -TotalCount 1) }
}
if ($rows.Count -gt 0) {
    $env:HARNESS_MODE = "serve"
    Write-Host "  逐行输入官方 validation 样本，逐行输出预测（A/B/C 自动识别）："
    $rows | python -u harness/entry.py --echo-task | ForEach-Object {
        $_.Substring(0, [Math]::Min(150, $_.Length))
    }
}

$modelRoot = Join-Path $demoOut "runs"
$manifest = Get-ChildItem $modelRoot -Recurse -Filter manifest.json -ErrorAction SilentlyContinue |
            Select-Object -First 1
if ($manifest) {
    Write-Host ""
    Write-Host "  交付产物审计（$($manifest.FullName)）："
    Get-Content $manifest.FullName -Raw
}

# ---------------------------------------------------------------- 5. 记录
Section "5 · 实验记录与 Sampler 微调（任务 2 要求）"
$sim = Get-ChildItem runs -Filter "sim_rounds_*.json" -ErrorAction SilentlyContinue |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($sim) { Write-Host "  本地协议复刻记录：$($sim.Name)" }
$sft = "runs\trio_sft_report.json"
if (Test-Path $sft) {
    $r = Get-Content $sft -Raw | ConvertFrom-Json
    Write-Host "  Sampler 微调：base=$($r.base_model)  LoRA rank=$($r.lora_rank)  步数=$($r.steps)"
    Write-Host "  损失：$($r.loss_first) -> $($r.loss_last)（44 步，4 epoch）"
    Write-Host "  权重路径：$($r.sampler_path)"
}
Write-Host "  SwanLab 实验记录：https://swanlab.cn/@zhaowenbo/zhixun-engine-3tasks"
Write-Host "  源码仓库：https://github.com/leather151/zhixun-harness"

Section "6 · 演示结束"
Write-Host "  本次演示产物目录：$demoOut"
Write-Host "  闭环逐轮证据：windows.jsonl / decisions.jsonl / report.md / manifest.json"
Write-Host "  说明：以上全部在本机真实运行；隐藏测试集评测由平台独立完成，本地不可得。"

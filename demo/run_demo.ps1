<#
  演示视频一键执行脚本（目标 < 4 分钟，留出解说余量）

  用法（工程根目录）：
      .\demo\run_demo.ps1                 # 完整演示
      .\demo\run_demo.ps1 -SkipLoop       # 跳过闭环节
      .\demo\run_demo.ps1 -Offline        # 无网络时：策略回落规则版，其余照跑

  这一版对应的三个关键改进（都按赛事规则要求）：
  1) 策略层真的调用 PyTRiO 的 **Qwen/Qwen3.8-27B**（规则点名的模型），逐窗打印决策与 token 用量；
  2) **三轮独立实验**：每轮一个假设 -> 改配方真训 -> 按公开验证集决定采纳/驳回；
  3) **新数据包兜底**：造一个与 A/B/C 完全不同的 7 维数据包，仍能训练+预测（评测换数据集也不崩）。
#>
param(
    [string]$Task = "B",
    [int]$Windows = 4,
    [switch]$SkipLoop,
    [switch]$Offline
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Get-Location).Path
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

Section "0 · 智训引擎 · 工业模型自进化系统"
Write-Host "  数据：UCI Gas / XJTU-SY 轴承 / NASA C-MAPSS（官方公开部分）"
Write-Host "  闭环：漂移监测 -> 诊断归因 -> 实验优化 -> 评估上线 -> 回滚"
Write-Host "  策略层：PyTRiO Qwen/Qwen3.8-27B（规则兜底）；模型层：CPU 基线自迭代"
Pause-Brief 1200

Section "1 · 镜像自检：三任务代码 / 模型 / 数据可达性 + 真实样本推理"
$env:HARNESS_MODE = "selftest"
python -u harness/entry.py
Pause-Brief 700

if (-not $SkipLoop) {
    Section "2 · 闭环实跑：Qwen3.8-27B 逐窗决策（任务 $Task）"
    Write-Host "  每窗打印：指标 / 参照 / 漂移强度 / 决策动作 / 是否上线"
    $env:HARNESS_MODE = "loop"
    $policy = if ($Offline) { "rule" } else { "llm" }
    python -u harness/entry.py --task $Task --stream train --windows $Windows `
        --policy $policy --swan off --out (Join-Path $demoOut "runs")
    Pause-Brief 700
}

Section "3 · 三轮独立实验：假设 -> 改配方 -> 真训 -> 采纳/驳回（任务 A）"
Write-Host "  调用官方契约 python -m baselines.train；每轮换一个假设"
python -u demo/round_protocol_sim.py --task A --rounds 3 --engine experiment
Pause-Brief 700

Section "4 · 新数据包兜底：7 维全新 schema 也能训练与预测"
Write-Host "  造一个与 A/B/C 都不同的包（字段 temp/vib/curr/... ，4 类）"
python -u demo/make_new_package.py
$pkg = Join-Path $env:TEMP "newpkg"
Remove-Item (Join-Path $pkg "round-1") -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "  用官方契约训练（自动切换自适应通道，不使用任何预训练权重）："
python -m baselines.train --profile A --train "$pkg\train.jsonl" --validation "$pkg\validation.jsonl" `
    --output "$pkg\round-1" --epochs 20 --stage-id new-pkg-1 --json *> "$pkg\train.log"
$rep = Get-Content "$pkg\round-1\training.json" -Raw -ErrorAction SilentlyContinue
if ($rep) {
    $r = $rep | ConvertFrom-Json
    Write-Host "  维度=$($r.input_features)  隐藏=$($r.hidden)  参数=$($r.parameter_count)  步数=$($r.final_steps)"
    Write-Host "  验证准确率 $($r.validation_before.accuracy) -> $($r.validation_after.accuracy)"
}
Write-Host ""
Write-Host "  按 schema 找模型并预测（不传任务名）："
$env:HARNESS_MODELS = $pkg
python -c "import sys,json; sys.path.insert(0,'.'); from baselines.predict import predict_auto; rows=[json.loads(l) for l in open(r'$pkg\validation.jsonl',encoding='utf-8')]; [print('   真值=%-7s 预测=%-7s'%(r['answer'], max(predict_auto(r['inputs'])['outputs'], key=lambda o:o['score'])['label'])) for r in rows[:4]]"
Remove-Item Env:HARNESS_MODELS
Pause-Brief 700

Section "5 · 官方 predict 协议 + 交付产物审计"
$rows = @()
$base3 = "D:\人工智能大赛\02_数据"
foreach ($d in @("A_气体传感器", "B_滚动轴承RUL", "C_涡扇发动机RUL")) {
    $p = Join-Path $base3 "$d\validation.jsonl"
    if (Test-Path $p) { $rows += (Get-Content $p -TotalCount 1) }
}
if ($rows.Count -gt 0) {
    $env:HARNESS_MODE = "serve"
    Write-Host "  stdin 一行请求 / stdout 一行结果，A/B/C 自动识别："
    $rows | python -u harness/entry.py --echo-task |
        ForEach-Object { $_.Substring(0, [Math]::Min(140, $_.Length)) }
}
$manifest = Get-ChildItem (Join-Path $demoOut "runs") -Recurse -Filter manifest.json -ErrorAction SilentlyContinue |
            Select-Object -First 1
if ($manifest) {
    Write-Host ""
    Write-Host "  交付产物 manifest（sha256 / 父检查点 / 逐窗指标 / 成本）："
    Get-Content $manifest.FullName -Raw
}

Section "6 · 实验记录与 Sampler 微调"
$sft = "runs\trio_sft_report.json"
if (Test-Path $sft) {
    $r = Get-Content $sft -Raw | ConvertFrom-Json
    Write-Host "  Sampler 微调：$($r.base_model) / LoRA rank $($r.lora_rank) / $($r.steps) 步"
    Write-Host "  损失 $($r.loss_first) -> $($r.loss_last)；权重 $($r.sampler_path)"
}
Write-Host "  SwanLab：https://swanlab.cn/@zhaowenbo/zhixun-engine-3tasks"
Write-Host "  源码：https://github.com/leather151/zhixun-harness"

Section "7 · 演示结束"
Write-Host "  产物目录：$demoOut"
Write-Host "  说明：以上均为本机真实运行；隐藏测试集评测由平台独立完成，本地不可得。"

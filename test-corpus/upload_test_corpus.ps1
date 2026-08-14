<#
.SYNOPSIS
    将 docs/test-corpus 下的 .md 测试制度文档批量上传到知识库。

.DESCRIPTION
    默认把 6 份文档灌进一个「company 公开知识库」，全员（含普通账号）都能检索到，
    适合多账号 / 权限测试。也支持 -Scope personal 灌个人库（仅上传账号可见）。

    流程：登录拿 JWT -> （company/dept 时）幂等地建库或复用同名库 -> 逐个上传。
    兼容 Windows PowerShell 5.1（手动构造 multipart/form-data，不依赖 6+ 的 -Form）。

    权限要点：
    - company 库仅「总管理员」可创建，故 -Scope company 时必须用管理员账号登录。
    - 检索时：管理员全库可见；普通用户可见 = 自己的个人库 + 本部门 dept 库 + company 公开库 + 被授权库。

.PARAMETER Username   登录用户名（-Scope company 时必须是总管理员）。
.PARAMETER Password   登录密码。
.PARAMETER Scope      company（默认，公开库）| personal（个人库）。
.PARAMETER KbName     目标知识库名称（company 时用于幂等查找/创建），默认「测试制度库（行政）」。
.PARAMETER ApiBase    后端 FastAPI 地址，默认 http://127.0.0.1:8000（并存时改 8010）。
.PARAMETER UserBase   Django 用户服务地址，默认 http://127.0.0.1:8001。

.EXAMPLE
    .\upload_test_corpus.ps1 -Username admin -Password 123456
.EXAMPLE
    .\upload_test_corpus.ps1 -Username admin -Password 123456 -Scope personal
#>
param(
    [Parameter(Mandatory = $true)] [string]$Username,
    [Parameter(Mandatory = $true)] [string]$Password,
    [ValidateSet("company", "personal")] [string]$Scope = "company",
    [string]$KbName   = "测试制度库（行政）",
    [string]$ApiBase  = "http://127.0.0.1:8000",
    [string]$UserBase = "http://127.0.0.1:8001"
)

$ErrorActionPreference = "Stop"

# 绕过系统代理（本机直连，避免代理拦截 localhost；与项目 runbook 一致）
$env:HTTP_PROXY = ""; $env:HTTPS_PROXY = ""; $env:http_proxy = ""; $env:https_proxy = ""
[System.Net.WebRequest]::DefaultWebProxy = $null

$corpusDir = $PSScriptRoot
Write-Host "[1/4] 语料目录: $corpusDir  (Scope=$Scope)"

# ── 登录拿 token ──────────────────────────────────────────────
Write-Host "[2/4] 登录 $UserBase/user/login/ ..."
try {
    $loginBody = @{ username = $Username; password = $Password } | ConvertTo-Json
    $loginResp = Invoke-RestMethod -Uri "$UserBase/user/login/" -Method Post `
        -ContentType "application/json" -Body $loginBody -UseBasicParsing
} catch {
    Write-Error "登录失败：$($_.Exception.Message)"; exit 1
}
$token = $loginResp.token
if (-not $token) { Write-Error "登录响应里没有 token，请检查账号密码。"; exit 1 }
$authHeader = @{ Authorization = "Bearer $token" }
Write-Host "      登录成功，已获取 token。"

# ── 确定上传目标（company：幂等建库/复用；personal：个人库）──
$kbId = $null
if ($Scope -eq "company") {
    Write-Host "[3/4] 确保 company 公开库「$KbName」存在 ..."
    $list = Invoke-RestMethod -Uri "$ApiBase/api/kb/list" -Method Get -Headers $authHeader -UseBasicParsing
    $existing = $list.data.kbs | Where-Object { $_.scope -eq "company" -and $_.name -eq $KbName } | Select-Object -First 1
    if ($existing) {
        $kbId = $existing.kb_id
        Write-Host "      已存在，复用 kb_id=$kbId"
    } else {
        $createBody = @{ name = $KbName; scope = "company"; description = "测试用企业行政制度语料" } | ConvertTo-Json
        try {
            $created = Invoke-RestMethod -Uri "$ApiBase/api/kb" -Method Post -Headers $authHeader `
                -ContentType "application/json" -Body $createBody -UseBasicParsing
        } catch {
            $msg = $_.Exception.Message; if ($_.ErrorDetails.Message) { $msg = $_.ErrorDetails.Message }
            Write-Error "创建 company 库失败（company 库仅总管理员可建，请确认账号权限）：$msg"; exit 1
        }
        $kbId = $created.data.kb_id
        Write-Host "      已创建，kb_id=$kbId"
    }
    $uploadUri = "$ApiBase/api/kb/$kbId/documents"
} else {
    Write-Host "[3/4] 目标：个人文档库（仅上传账号可检索到）"
    $uploadUri = "$ApiBase/api/vector/add/single"
}

# ── multipart 上传函数（5.1 兼容：按字节拼 body，中文文件名用 UTF-8）──
function Send-OneFile {
    param([string]$Uri, [hashtable]$Headers, [string]$FilePath)
    $boundary  = [System.Guid]::NewGuid().ToString()
    $fileName  = [System.IO.Path]::GetFileName($FilePath)
    $fileBytes = [System.IO.File]::ReadAllBytes($FilePath)
    $enc = [System.Text.Encoding]::UTF8
    $nl  = "`r`n"

    $header = "--$boundary$nl" +
              "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"$nl" +
              "Content-Type: text/markdown$nl$nl"
    $footer = "$nl--$boundary--$nl"

    $ms = New-Object System.IO.MemoryStream
    $hb = $enc.GetBytes($header); $fb = $enc.GetBytes($footer)
    $ms.Write($hb, 0, $hb.Length); $ms.Write($fileBytes, 0, $fileBytes.Length); $ms.Write($fb, 0, $fb.Length)
    $bodyBytes = $ms.ToArray(); $ms.Dispose()

    return Invoke-RestMethod -Uri $Uri -Method Post -UseBasicParsing `
        -ContentType "multipart/form-data; boundary=$boundary" -Headers $Headers -Body $bodyBytes
}

# ── 逐个上传 .md ──────────────────────────────────────────────
$files = Get-ChildItem -Path $corpusDir -Filter "*.md" | Sort-Object Name
Write-Host "[4/4] 待上传 $($files.Count) 个文档 -> $uploadUri`n"

$ok = 0; $fail = 0
foreach ($f in $files) {
    try {
        $resp = Send-OneFile -Uri $uploadUri -Headers $authHeader -FilePath $f.FullName
        Write-Host ("  [OK]   {0}  ->  {1}" -f $f.Name, $resp.message)
        $ok++
    } catch {
        $detail = $_.Exception.Message; if ($_.ErrorDetails.Message) { $detail = $_.ErrorDetails.Message }
        Write-Host ("  [FAIL] {0}  ->  {1}" -f $f.Name, $detail) -ForegroundColor Yellow
        $fail++
    }
}

Write-Host "`n完成：成功 $ok，失败 $fail。Scope=$Scope$(if($kbId){" kb_id=$kbId"})"
Write-Host "提示：AI 聊天页需后端 AGENT_ENGINE=graph；按 README 测试问题清单逐项验证。"

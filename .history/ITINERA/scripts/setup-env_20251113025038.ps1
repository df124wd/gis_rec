param(
  [string]$ConfigPath = "$PSScriptRoot/../config/app_config.json"
)
if (!(Test-Path $ConfigPath)) { Write-Error "config not found: $ConfigPath"; exit 1 }
$json = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$keys = @('OPENAI_BASE_URL','OPENAI_API_KEY','DEEPSEEK_API_KEY','OPENAI_CHAT_MODEL','OPENAI_EMBEDDING_MODEL','TIANDITU_TK','AMAP_KEY')
foreach ($k in $keys) {
  $v = ($json.$k) -as [string]
  if ($v) {
    $v = $v.Trim()
    [Environment]::SetEnvironmentVariable($k,$v,'Process')
    if ($k -match 'KEY') { Write-Output ("{0} set" -f $k) } else { Write-Output ("{0}={1}" -f $k,$v) }
  }
}
Write-Output "Environment variables applied for current session."
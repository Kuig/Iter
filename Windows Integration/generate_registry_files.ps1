# Determine paths dynamically
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = (Get-Item $ScriptDir).Parent.FullName
$IterPath = Join-Path $ProjectDir ".venv\Scripts\iter.exe"

# Format paths for Windows Registry
$EscapedIterPath = $IterPath.Replace("\", "\\")

# Scan the Presets directory for existing preset JSON files
$PresetsDir = Join-Path $ProjectDir "Presets"
if (-not (Test-Path $PresetsDir)) {
    New-Item -ItemType Directory -Path $PresetsDir -Force | Out-Null
}

$Presets = Get-ChildItem -Path $PresetsDir -Filter "*.json" | Sort-Object Name

# Generate register.reg
$RegisterContent = "Windows Registry Editor Version 5.00`r`n"
# Generate unregister.reg
$UnregisterContent = "Windows Registry Editor Version 5.00`r`n"

$SupportedExtensions = @(".md", ".markdown")

foreach ($Ext in $SupportedExtensions) {
    # Write the main entry for this extension to add the "Process with Iter..." cascading menu
    $RegisterContent += "`r`n[HKEY_CLASSES_ROOT\SystemFileAssociations\$Ext\shell\Iter]`r`n"
    $RegisterContent += "`"MUIVerb`"=generic:`"Process with Iter...`"`r`n".Replace("generic:", "")
    $RegisterContent += "`"SubCommands`"=generic:`"`"`r`n".Replace("generic:", "")

    # Write the unregister clean key for this extension
    $UnregisterContent += "`r`n[-HKEY_CLASSES_ROOT\SystemFileAssociations\$Ext\shell\Iter]`r`n"
    
    $Counter = 1
    foreach ($Preset in $Presets) {
        $PresetName = $Preset.BaseName
        $Key = "{0:D2}_{1}" -f $Counter, $PresetName
        $Verb = $PresetName
        
        # Registry command that runs iter.exe and keeps the console open with pause so the user can see results
        $CommandString = 'cmd.exe /c ""' + $EscapedIterPath + '" run --input "%1" --output "%~dpn1_processed%~x1" --preset "' + $PresetName + '" & pause"'
        
        $RegisterContent += "`r`n[HKEY_CLASSES_ROOT\SystemFileAssociations\$Ext\shell\Iter\shell\$Key]`r`n"
        $RegisterContent += "`"MUIVerb`"=generic:`"$Verb`"`r`n".Replace("generic:", "")
        
        $RegisterContent += "`r`n[HKEY_CLASSES_ROOT\SystemFileAssociations\$Ext\shell\Iter\shell\$Key\command]`r`n"
        $RegisterContent += '@="' + $CommandString.Replace('\', '\\').Replace('"', '\"') + '"' + "`r`n"
        
        $Counter++
    }
}

# Output registry files as UTF-16 LE (Unicode) for native Windows compatibility
$RegisterPath = Join-Path $ScriptDir "register.reg"
$UnregisterPath = Join-Path $ScriptDir "unregister.reg"

$RegisterContent | Out-File -FilePath $RegisterPath -Encoding Unicode
$UnregisterContent | Out-File -FilePath $UnregisterPath -Encoding Unicode

Write-Host "Generated register.reg and unregister.reg successfully in:"
Write-Host "   $ScriptDir"
if ($Presets.Count -eq 0) {
    Write-Host "Warning: No presets found in Presets/. The context menu will be empty until you add preset files and rerun this script."
} else {
    $NamesStr = $Presets.BaseName -join ', '
    Write-Host "Presets included: $NamesStr"
}

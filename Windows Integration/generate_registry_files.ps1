# Determine paths dynamically
$ScriptDir = $PSScriptRoot
$ProjectDir = (Get-Item $ScriptDir).Parent.FullName
$IterPath = Join-Path $ProjectDir ".venv\Scripts\iter.exe"
$RunPath = Join-Path $ScriptDir "run_preset.cmd"
$GuiPath = Join-Path $ScriptDir "launch_gui.bat"

if (-not (Test-Path $IterPath)) {
    Write-Warning "$IterPath not found: create the .venv and install the project before using the menu."
}

# Every preset entry runs run_preset.cmd, which calls "iter run" from the project folder
# and writes "<name>_processed<ext>" next to the source file.
# "Open with Iter" runs launch_gui.bat, which opens the GUI with the file as its input.
$EscapedRunPath = $RunPath.Replace("\", "\\")
$EscapedGuiPath = $GuiPath.Replace("\", "\\")

# Scan the Presets directory for existing preset JSON files
$PresetsDir = Join-Path $ProjectDir "Presets"
if (-not (Test-Path $PresetsDir)) {
    New-Item -ItemType Directory -Path $PresetsDir -Force | Out-Null
}

$Presets = @(Get-ChildItem -Path $PresetsDir -Filter "*.json" | Sort-Object Name)

# Explorer shows at most 16 entries of a menu, submenus included, and drops the rest
# without a word.
$MaxEntries = 16
if ($Presets.Count -gt $MaxEntries) {
    throw "Presets/ has $($Presets.Count) presets: Explorer shows only the first $MaxEntries."
}

# Generate register.reg
$RegisterContent = "Windows Registry Editor Version 5.00`r`n"
# Generate unregister.reg
$UnregisterContent = "Windows Registry Editor Version 5.00`r`n"

$SupportedExtensions = @(".md", ".markdown")

foreach ($Ext in $SupportedExtensions) {
    $Shell = "HKEY_CLASSES_ROOT\SystemFileAssociations\$Ext\shell"

    # Remove the entries of an earlier registration, so deleted or renamed presets disappear,
    # then write the "Process with Iter..." cascading menu
    $RegisterContent += "`r`n[-$Shell\Iter]`r`n"
    $RegisterContent += "`r`n[$Shell\Iter]`r`n"
    $RegisterContent += "`"MUIVerb`"=`"Process with Iter...`"`r`n"
    $RegisterContent += "`"SubCommands`"=`"`"`r`n"

    $Counter = 1
    foreach ($Preset in $Presets) {
        $PresetName = $Preset.BaseName
        $Key = "{0:D2}_{1}" -f $Counter, $PresetName

        $RegisterContent += "`r`n[$Shell\Iter\shell\$Key]`r`n"
        $RegisterContent += "`"MUIVerb`"=`"$PresetName`"`r`n"
        $RegisterContent += "`r`n[$Shell\Iter\shell\$Key\command]`r`n"
        $RegisterContent += '@="\"' + $EscapedRunPath + '\" \"%1\" \"' + $PresetName + '\""' + "`r`n"

        $Counter++
    }

    # The entry that opens the GUI on the file
    $RegisterContent += "`r`n[-$Shell\IterGui]`r`n"
    $RegisterContent += "`r`n[$Shell\IterGui]`r`n`"MUIVerb`"=`"Open with Iter`"`r`n"
    $RegisterContent += "`r`n[$Shell\IterGui\command]`r`n"
    $RegisterContent += '@="\"' + $EscapedGuiPath + '\" \"%1\""' + "`r`n"

    # Write the unregister clean keys for this extension
    $UnregisterContent += "`r`n[-$Shell\Iter]`r`n"
    $UnregisterContent += "`r`n[-$Shell\IterGui]`r`n"
}

# Output registry files as UTF-16 LE (Unicode) for native Windows compatibility
$RegisterPath = Join-Path $ScriptDir "register.reg"
$UnregisterPath = Join-Path $ScriptDir "unregister.reg"

$RegisterContent | Out-File -FilePath $RegisterPath -Encoding Unicode
$UnregisterContent | Out-File -FilePath $UnregisterPath -Encoding Unicode

Write-Host "Generated register.reg and unregister.reg successfully in:"
Write-Host "   $ScriptDir"
if ($Presets.Count -eq 0) {
    Write-Host "Warning: No presets found in Presets/. The preset submenu will be empty until you add preset files and rerun this script."
} else {
    $NamesStr = $Presets.BaseName -join ', '
    Write-Host "Presets included: $NamesStr"
}

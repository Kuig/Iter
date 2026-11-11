@echo off
REM Called by the Explorer context menu entries (see generate_registry_files.ps1):
REM   run_preset.cmd <source file or folder> <preset name>
REM Runs the preset on the source through the project's CLI, which writes its default
REM output: <name>_processed<ext> next to a file; for a folder, <folder>_processed
REM beside it (mirror layout) or <folder>_processed.md (merged layout). The name is
REM fixed, unlike TextConverter's numbered outputs: the progress file is named after
REM the output, so an interrupted run resumes only if the next one writes to the same
REM place. The CLI runs from the project folder, where Presets\, config.json and
REM secrets.json are. On failure the window stays open so the error can be read.
setlocal
set "SOURCE=%~f1"
set "PRESET=%~2"
cd /d "%~dp0.."
set "EXE=%CD%\.venv\Scripts\iter.exe"

if not exist "%EXE%" (
    echo Error: %EXE% not found. Create the virtual environment and install the project first.
    pause
    exit /b 1
)
"%EXE%" run --input "%SOURCE%" --preset "%PRESET%"
if errorlevel 1 pause

<#
    CourseMind setup (Windows / PowerShell).

    Creates the virtual environment with the RIGHT Python and installs the
    dependencies. Run from the project folder:

        .\setup.ps1

    Add -Force to rebuild an existing venv from scratch.

    Why this script exists: running a bare `python -m venv venv` picks up
    whatever `python` happens to be first on PATH. On this project that has
    twice produced a broken environment - once an MSYS2/mingw Python (which
    creates a Unix-layout venv with bin/ instead of Scripts/) and once the
    free-threaded python3.14t.exe (which has no wheels for chromadb /
    tiktoken / pydantic-core, so the install dies half-way and leaves
    streamlit installed but pypdf missing).
#>
param([switch]$Force)

$ErrorActionPreference = "Stop"
$RequiredMajorMinor = "3.12"

Write-Host "CourseMind setup" -ForegroundColor Cyan
Write-Host ""

# --- 1. locate Python 3.12 -------------------------------------------------
$python = $null
try {
    $candidate = (& py "-$RequiredMajorMinor" -c "import sys; print(sys.executable)" 2>$null)
    if ($LASTEXITCODE -eq 0 -and $candidate) { $python = $candidate.Trim() }
} catch { }

if (-not $python) {
    $fallback = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path $fallback) { $python = $fallback }
}

if (-not $python) {
    Write-Host "Python $RequiredMajorMinor was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/release/python-3120/ and run this again."
    Write-Host "(3.13 and 3.14 are NOT usable here: chromadb / tiktoken / pydantic-core have no wheels for them yet.)"
    exit 1
}
Write-Host "Using: $python"

# --- 2. create the venv ----------------------------------------------------
if (Test-Path "venv") {
    if ($Force) {
        Write-Host "Removing the existing venv (-Force)..."
        Remove-Item -Recurse -Force "venv"
    } else {
        Write-Host "A 'venv' folder already exists. Re-run with -Force to rebuild it." -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "Creating the virtual environment..."
& $python -m venv venv
if ($LASTEXITCODE -ne 0) { Write-Host "venv creation failed." -ForegroundColor Red; exit 1 }

$venvPython = ".\venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "venv\Scripts\python.exe is missing - the Python used was not a standard Windows build." -ForegroundColor Red
    exit 1
}
$version = (& $venvPython --version)
Write-Host "Virtual environment: $version"

# --- 3. dependencies -------------------------------------------------------
Write-Host "Installing dependencies (this takes a few minutes)..."
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Dependency installation failed." -ForegroundColor Red; exit 1 }

# --- 4. verify what actually matters --------------------------------------
Write-Host ""
Write-Host "Verifying the install..."
$missing = @()
foreach ($pkg in @("streamlit", "pypdf", "chromadb", "crewai", "openai", "tiktoken", "dotenv")) {
    & $venvPython -c "import $pkg" 2>$null
    if ($LASTEXITCODE -ne 0) { $missing += $pkg }
}
if ($missing.Count -gt 0) {
    Write-Host "These packages did not import: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "All required packages import correctly." -ForegroundColor Green

# --- 5. .env ---------------------------------------------------------------
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") { Copy-Item ".env.example" ".env" }
    Write-Host ""
    Write-Host "Next step: put your OpenAI API key in the .env file." -ForegroundColor Yellow
    Write-Host "  notepad .env"
}

Write-Host ""
Write-Host "Done. Start the app with:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\python.exe -m streamlit run app.py"

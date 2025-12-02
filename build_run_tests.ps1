[CmdletBinding()]
param(
    [int]$Index = 0
)

Write-Host "Build (if needed) and run tests"
$srcDir = "src"
$exe = "pauli_sim.exe"

# Compile
Write-Host "Compiling $exe..."
g++ -DCPU_ONLY -std=c++17 -Wall -O2 "$srcDir/pauli.cpp" "$srcDir/main.cpp" "$srcDir/tests.cpp" -I"$srcDir" -o "$exe"
if ($LASTEXITCODE -ne 0) { Write-Error "Compilation failed."; exit $LASTEXITCODE }

if ($Index -gt 0) {
    Write-Host "Running single test index $Index"
    & .\$exe -d cpu -i $Index
} else {
    Write-Host "Running all tests"
    & .\$exe -d cpu
}

Write-Host "Done."
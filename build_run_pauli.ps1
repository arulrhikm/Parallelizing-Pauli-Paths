Write-Host "Build and run pauli simulator (CPU-only)"

$srcDir = "src"
$exe = "pauli_sim.exe"

# Compile main simulator
g++ -DCPU_ONLY -std=c++17 -Wall -O2 "$srcDir/pauli.cpp" "$srcDir/main.cpp" "$srcDir/tests.cpp" -I"$srcDir" -o "$exe"
if ($LASTEXITCODE -ne 0) { Write-Error "Compilation failed."; exit $LASTEXITCODE }

# Run all tests using CPU simulator
Write-Host "Running $exe -d cpu"
& .\$exe -d cpu

Write-Host "Done."
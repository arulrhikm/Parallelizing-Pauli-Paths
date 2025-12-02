Write-Host "Build and run diagnostic (CPU-only)"

$srcDir = "src"
$diag = "diagnostic.exe"

g++ -DCPU_ONLY -std=c++17 -Wall -O2 "$srcDir/diagnostic.cpp" "$srcDir/pauli.cpp" -I"$srcDir" -o "$diag"
if ($LASTEXITCODE -ne 0) { Write-Error "Compilation failed."; exit $LASTEXITCODE }

Write-Host "Running $diag..."
& .\$diag

Write-Host "Done."
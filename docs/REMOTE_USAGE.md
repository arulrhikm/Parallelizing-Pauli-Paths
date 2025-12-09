# Using Python Scripts on Remote Server

## Quick Guide for Remote Execution

### 1. SSH to Remote Server
```bash
ssh arulm@ghc27.ghc.andrew.cmu.edu
cd ~/private/15418/Parallelizing-Pauli-Paths
```

### 2. Install matplotlib (if needed)
```bash
pip install --user matplotlib
# Or if you have sudo:
sudo pip install matplotlib
```

### 3. Run Scripts
```bash
# Generate all figures
python3 generate_all_figures.py

# Or run individually
python3 performance_analysis.py
python3 algorithmic_visualization.py
python3 correctness_validation.py
python3 interactive_demo.py
```

### 4. View Generated Files
```bash
# List all PNG files
ls -lh *.png

# Check file sizes
du -h *.png
```

### 5. Copy Plots to Local Machine

**From your LOCAL machine (not on server):**

```bash
# Copy all PNG files
scp arulm@ghc27.ghc.andrew.cmu.edu:~/private/15418/Parallelizing-Pauli-Paths/*.png ./

# Or copy to specific directory
scp arulm@ghc27.ghc.andrew.cmu.edu:~/private/15418/Parallelizing-Pauli-Paths/*.png ~/Downloads/

# Copy JSON data files too
scp arulm@ghc27.ghc.andrew.cmu.edu:~/private/15418/Parallelizing-Pauli-Paths/*.json ./
```

**Or use the provided script (from local machine):**
```bash
chmod +x copy_plots.sh
./copy_plots.sh
```

## Generated Files

After running scripts, you'll have:

**PNG Plots:**
- `performance_analysis.png` - 4-panel speedup analysis
- `parameter_analysis.png` - Parameter sensitivity
- `clifford_analysis.png` - Gate type comparison
- `memory_analysis.png` - Memory patterns
- `pauli_evolution.png` - Word dynamics
- `correctness_validation.png` - Test results
- `timing_comparison.png` - CPU vs GPU timing
- `speedup_chart.png` - Speedup factors
- `performance_scaling.png` - Log-scale performance
- `report_summary.png` - 4-panel summary

**Data Files:**
- `validation_results.json` - Detailed validation data

## Troubleshooting

**"matplotlib not found"**
```bash
pip install --user matplotlib
```

**"No module named numpy"**
```bash
pip install --user numpy matplotlib
```

**Plots not generating**
- Check that matplotlib is installed: `python3 -c "import matplotlib; print('OK')"`
- Check that scripts completed without errors
- Verify PNG files were created: `ls *.png`

**Can't view plots on server**
- This is normal - remote servers don't have display
- Copy plots to local machine using `scp` (see above)
- Open PNG files on your local machine with any image viewer

## Complete Workflow

```bash
# 1. On remote server
ssh arulm@ghc27.ghc.andrew.cmu.edu
cd ~/private/15418/Parallelizing-Pauli-Paths
python3 generate_all_figures.py

# 2. On local machine (new terminal)
scp arulm@ghc27.ghc.andrew.cmu.edu:~/private/15418/Parallelizing-Pauli-Paths/*.png ./

# 3. View plots locally
open *.png  # macOS
xdg-open *.png  # Linux
# Or just double-click PNG files in file manager
```


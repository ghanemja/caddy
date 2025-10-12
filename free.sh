mkdir -p ~/bin
cat > ~/bin/FreeCADCmd <<'SH'
#!/usr/bin/env bash
# Minimal FreeCADCmd wrapper that uses the FreeCAD libs from the active conda env
set -euo pipefail

# Prefer the active conda env; fall back to $HOME/miniforge3/envs/freecad
CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniforge3/envs/freecad}"

# Make sure FreeCAD’s libs are visible
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages:${PYTHONPATH:-}"

if [[ "${1:-}" == "--version" ]]; then
  python - <<'PY'
import FreeCAD
print(FreeCAD.Version())
PY
  exit 0
fi

# If a script was passed, run it; otherwise drop to an interactive Python with FreeCAD loaded
if [[ -n "${1:-}" && -f "$1" ]]; then
  python "$@"
else
  python - <<'PY'
import FreeCAD; print("FreeCAD ready:", FreeCAD.Version())
print("Tip: run `FreeCADCmd my_script.py` to execute a script.")
PY
fi
SH
chmod +x ~/bin/FreeCADCmd
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# test it
FreeCADCmd --version
FreeCADCmd -c "print('hello via FreeCADCmd wrapper')"

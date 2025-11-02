"""Setup rápido para notebooks del proyecto 01-toolkit-analisis-bursatil."""

import sys
from pathlib import Path

# Asegurar que la carpeta notebooks esté en sys.path
notebooks_path = Path(__file__).parent.resolve()
if str(notebooks_path) not in sys.path:
    sys.path.insert(0, str(notebooks_path))

def setup_notebook():
    """Configura entorno de ejecución en notebooks."""
    # --- Detectar raíz del proyecto ---
    cwd = Path.cwd().resolve()
    project_root = cwd if (cwd / "src").exists() else cwd.parent
    src_path = project_root / "src"

    # --- Añadir src al sys.path ---
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Añadido al sys.path: {src_path}")
    else:
        print("ℹ️  src ya estaba en sys.path")

    # --- Activar autoreload (solo si estás en Jupyter) ---
    try:
        ipython = get_ipython()  # type: ignore
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
        print("♻️  Autoreload activado")
    except Exception:
        print("⚠️  No se pudo activar autoreload (no estás en Jupyter).")

    # --- Diagnóstico ---
    print(f"📁 Proyecto raíz: {project_root}")
    print(f"📂 Directorio de trabajo actual: {cwd}")
    print("✅ Notebook listo para usar clases del proyecto")

    return {"project_root": project_root, "src_path": src_path, "cwd": cwd}


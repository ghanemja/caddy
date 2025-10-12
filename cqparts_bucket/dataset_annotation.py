# annotate.py
# Interactive dataset builder for before/after CAD edits

import os
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import importlib.util
from cadquery import exporters
import cadquery as cq

SAMPLES_DIR = "samples"
RENDER_FILENAME = "render.png"

os.makedirs(SAMPLES_DIR, exist_ok=True)

def import_cq_script(path):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_model(mod):
    if hasattr(mod, "make_model"):
        return mod.make_model()
    elif hasattr(mod, "make"):
        return mod.make()
    elif hasattr(mod, "build"):
        return mod.build()
    else:
        raise RuntimeError("No known entrypoint (make_model/make/build) found in script.")

def render_model(script_path, out_png):
    mod = import_cq_script(script_path)
    model = get_model(mod)
    # Export STL and render with show_object/exporters
    temp_stl = out_png.replace(".png", ".stl")
    exporters.export(model, temp_stl)
    try:
        import trimesh
        mesh = trimesh.load(temp_stl)
        scene = mesh.scene()
        scene.save_image(out_png, resolution=[640, 480])
    except Exception as e:
        print(f"[warn] Could not render {temp_stl}: {e}")
        os.rename(temp_stl, out_png)  # placeholder fallback

def annotate_pair(sample_id):
    sample_dir = os.path.join(SAMPLES_DIR, f"{sample_id:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    base_path = filedialog.askopenfilename(title="Select BASE model .py script")
    mod_path = filedialog.askopenfilename(title="Select MODIFIED model .py script")

    base_png = os.path.join(sample_dir, "base.png")
    mod_png = os.path.join(sample_dir, "mod.png")
    render_model(base_path, base_png)
    render_model(mod_path, mod_png)

    base_code = open(base_path).read()
    mod_code = open(mod_path).read()

    # Launch GUI
    root = tk.Tk()
    root.title("Annotate Edit Instruction")

    img_b = ImageTk.PhotoImage(Image.open(base_png))
    img_m = ImageTk.PhotoImage(Image.open(mod_png))

    tk.Label(root, text="BEFORE").grid(row=0, column=0)
    tk.Label(root, text="AFTER").grid(row=0, column=1)
    tk.Label(root, image=img_b).grid(row=1, column=0)
    tk.Label(root, image=img_m).grid(row=1, column=1)

    tk.Label(root, text="Instruction:").grid(row=2, column=0)
    entry = tk.Entry(root, width=80)
    entry.grid(row=2, column=1)

    def on_submit():
        instruction = entry.get().strip()
        if not instruction:
            print("[warn] Instruction is empty. Skipping.")
            root.destroy()
            return

        data = {
            "before_image": os.path.relpath(base_png, start=SAMPLES_DIR),
            "after_image": os.path.relpath(mod_png, start=SAMPLES_DIR),
            "instruction": instruction,
            "before_code": base_code,
            "after_code": mod_code
        }
        with open(os.path.join(sample_dir, "label.json"), "w") as f:
            json.dump(data, f, indent=2)
        print(f"[done] Saved sample {sample_id:05d}")
        root.destroy()

    tk.Button(root, text="Save Sample", command=on_submit).grid(row=3, column=0, columnspan=2)
    root.mainloop()

def build_dataset_jsonl():
    out_path = os.path.join(SAMPLES_DIR, "dataset.jsonl")
    with open(out_path, "w") as out:
        for name in sorted(os.listdir(SAMPLES_DIR)):
            sample = os.path.join(SAMPLES_DIR, name)
            if os.path.isdir(sample):
                label_path = os.path.join(sample, "label.json")
                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        j = json.load(f)
                    out.write(json.dumps(j) + "\n")
    print(f"[✓] Wrote dataset JSONL to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true", help="Launch GUI to annotate one sample")
    parser.add_argument("--build", action="store_true", help="Build dataset.jsonl from all annotations")
    parser.add_argument("--id", type=int, default=0, help="Sample ID to annotate (e.g., 1 for 00001)")
    args = parser.parse_args()

    if args.annotate:
        annotate_pair(args.id)
    if args.build:
        build_dataset_jsonl()

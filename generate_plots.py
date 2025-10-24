from framework.utils.plots_2 import plot_teacher_student_metrics
from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os


def main():
    CFG_PATH = f'experiments/Experiment_1/Run_1/config.toml'
    cfg = load_config_toml(CFG_PATH)
    setup_logging(cfg.model_dump(), Path(), False)

    for i in range(1,10):
        CFG_PATH = f'experiments/Experiment_1/Run_{i}/config.toml'
        plot_teacher_student_metrics(CFG_PATH)

    _make_overview_png(
        base_dir="experiments/Experiment_1",
        run_indices=range(1, 10),
        out_path="experiments/Experiment_1/overview_plot_Experiment_1.png",
    )
    # CFG_PATH = f'experiments/Experiment_3/Run_1/config.toml'
    # cfg = load_config_toml(CFG_PATH)
    # setup_logging(cfg.model_dump(), Path(), False)
    #
    # for i in range(1,25):
    #     CFG_PATH = f'experiments/Experiment_3/Run_{i}/config.toml'
    #     plot_teacher_student_metrics(CFG_PATH)
    #
    # _make_overview_png(
    #     base_dir="experiments/Experiment_3",
    #     run_indices=range(1, 25),
    #     out_path="experiments/Experiment_3/overview_plot_Experiment_3.png",
    # )

def _make_overview_png(
    base_dir="experiments/Experiment_3",
    run_indices=range(1, 25),
    hpad=40,               # padding between Loss and ValAcc columns
    vpad=60,               # padding between rows
    margin=60,             # outer border
    caption_width=400,     # left gutter for "Run i"
    out_path="experiments/Experiment_3/overview_all_plots.png",
    add_captions=True,
    final_scale=0.6,       # 1.0 = no final resize; <1.0 downscales once at the end
    set_png_dpi=400        # embeds DPI metadata (some viewers honor this)
):
    rows = []
    labels = []
    for i in run_indices:
        pdir = os.path.join(base_dir, f"Run_{i}", "plots")
        loss_p = os.path.join(pdir, "loss_per_epoch.png")
        acc_p  = os.path.join(pdir, "val_accuracy_per_epoch.png")
        if not (os.path.exists(loss_p) and os.path.exists(acc_p)):
            continue
        loss_im = Image.open(loss_p).convert("RGB")   # keep native size
        acc_im  = Image.open(acc_p).convert("RGB")

        # Equalize row height by padding the shorter one (no resampling)
        h1, h2 = loss_im.size[1], acc_im.size[1]
        row_h = max(h1, h2)
        def pad_to_h(img, target_h):
            if img.size[1] == target_h:
                return img
            padded = Image.new("RGB", (img.size[0], target_h), "white")
            padded.paste(img, (0, 0))
            return padded
        loss_im = pad_to_h(loss_im, row_h)
        acc_im  = pad_to_h(acc_im,  row_h)

        rows.append((loss_im, acc_im, row_h))
        labels.append(f"Run {i}")

    if not rows:
        raise FileNotFoundError("No runs with both plots found.")

    # Compute natural canvas size (no resizes)
    loss_w = rows[0][0].size[0]
    acc_w  = rows[0][1].size[0]
    col_w_total = loss_w + hpad + acc_w

    nrows = len(rows)
    canvas_w = margin + (caption_width if add_captions else 0) + col_w_total + margin
    canvas_h = margin + sum(rh for _, _, rh in rows) + vpad * (nrows - 1) + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except:
        font = ImageFont.load_default()
        font_bold = font

    # Column headers
    x0 = margin + (caption_width if add_captions else 0)
    header_y = max(10, margin // 2)
    draw.text((x0 + loss_w // 2 - 40, header_y), "Loss", font=font_bold, fill="black")
    draw.text((x0 + loss_w + hpad + acc_w // 2 - 60, header_y), "Val Acc", font=font_bold, fill="black")

    y = margin
    for (loss_im, acc_im, row_h), label in zip(rows, labels):
        if add_captions:
            cap_x = margin
            cap_y = y + row_h // 2 - 12
            draw.text((cap_x, cap_y), label, font=font_bold, fill="black")
        x = x0
        canvas.paste(loss_im, (x, y))
        x += loss_w + hpad
        canvas.paste(acc_im, (x, y))
        y += row_h + vpad

    # Optional single downscale at the end (best quality if needed)
    if final_scale != 1.0:
        new_w = int(canvas.size[0] * final_scale)
        new_h = int(canvas.size[1] * final_scale)
        canvas = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # PNG is lossless; compression level affects size, not quality.
    # Embed DPI metadata (viewers may use it for print scaling).
    canvas.save(out_path, format="PNG", optimize=True, compress_level=3, dpi=(set_png_dpi, set_png_dpi))
    print(f"Saved overview: {out_path}")


if __name__ == "__main__":
    main()

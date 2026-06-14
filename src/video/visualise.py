# ─────────────────────────────────────────────
# WHY this file exists:
# process.py saves timeline.json — a list of
# dicts with confidence/stress/engagement per
# timestamp. Numbers in a JSON file are hard to
# interpret at a glance. This file turns that
# JSON into a line chart so you can SEE how
# signals change over the video's duration.
# ─────────────────────────────────────────────

import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_timeline(path="data/samples/timeline.json"):
    # ─────────────────────────────────────────
    # WHY a separate load function?
    # Keeps "reading data" separate from "drawing
    # the chart" — same separation-of-concerns
    # pattern as analyse() vs print_report() earlier.
    # If you later load timeline data from a
    # database instead of a file, only this
    # function needs to change.
    # ─────────────────────────────────────────
    with open(path, "r") as f:
        return json.load(f)


def plot_timeline(timeline, output_path="data/samples/timeline_chart.png"):
    # ─────────────────────────────────────────
    # WHY filter out "no_person" entries first?
    # Frames where no person was detected have
    # confidence=None or missing keys entirely.
    # Plotting None values crashes matplotlib.
    # We build clean lists containing ONLY frames
    # where we have real data.
    # ─────────────────────────────────────────
    valid_entries = [e for e in timeline if not e.get("no_person", False)]

    if not valid_entries:
        print("❌ No valid frames to plot — every frame had no_person=True")
        return

    # ─────────────────────────────────────────
    # WHY list comprehensions for each signal?
    # We need FOUR separate lists for matplotlib:
    # - timestamps (x-axis, shared by all lines)
    # - confidence values (one line)
    # - stress values (another line)
    # - engagement values (third line)
    # Each comprehension pulls one field from
    # every entry into its own list.
    # ─────────────────────────────────────────
    timestamps = [e["timestamp"] for e in valid_entries]
    confidence = [e["confidence"] for e in valid_entries]
    stress     = [e["stress"] for e in valid_entries]
    engagement = [e["engagement"] for e in valid_entries]

    # ─────────────────────────────────────────
    # WHY plt.subplots(figsize=(10, 5))?
    # Creates a figure (the whole image) and an
    # axes object (the plotting area) in one call.
    # figsize is in inches — 10x5 gives a wide,
    # readable chart suitable for a report or demo.
    # ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    # ─────────────────────────────────────────
    # WHY three separate plot() calls?
    # Each call draws ONE line. marker='o' adds
    # a dot at each data point — useful since we
    # only have a few sampled frames, not a dense
    # continuous signal. label= sets the legend text.
    # ─────────────────────────────────────────
    ax.plot(timestamps, confidence, marker='o', label='Confidence', color='#1D9E75')
    ax.plot(timestamps, stress,     marker='o', label='Stress',     color='#D85A30')
    ax.plot(timestamps, engagement, marker='o', label='Engagement', color='#378ADD')

    # ─────────────────────────────────────────
    # WHY set y-axis limits to (0, 1)?
    # All three signals are normalised to 0.0-1.0
    # (thanks to normalise_scores()). Fixing the
    # y-axis range means the chart's scale is
    # consistent and comparable across different
    # videos — not auto-scaled to whatever the
    # min/max happens to be in THIS video.
    # ─────────────────────────────────────────
    ax.set_ylim(0, 1)

    # ─────────────────────────────────────────
    # WHY these labels and title?
    # Basic chart hygiene — anyone looking at this
    # image (without reading code) should understand
    # what it shows: time on x-axis, score on y-axis,
    # which line is which signal.
    # ─────────────────────────────────────────
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("Body Language Signals Over Time")

    # ─────────────────────────────────────────
    # WHY ax.legend()?
    # Without this, the three colored lines have
    # no labels visible on the chart itself — the
    # label= text we set in plot() only shows up
    # when legend() is called to render it.
    # ─────────────────────────────────────────
    ax.legend()

    # ─────────────────────────────────────────
    # WHY ax.grid(True, alpha=0.3)?
    # Adds light gridlines (30% opacity) to make
    # it easier to read exact values at a glance —
    # without overwhelming the chart with dark lines.
    # ─────────────────────────────────────────
    ax.grid(True, alpha=0.3)

    # ─────────────────────────────────────────
    # WHY plt.tight_layout()?
    # Without this, labels and titles can get cut
    # off at the edges of the saved image. This
    # automatically adjusts spacing so everything
    # fits cleanly within the figure bounds.
    # ─────────────────────────────────────────
    plt.tight_layout()

    # ─────────────────────────────────────────
    # WHY dpi=150?
    # dpi = dots per inch = image resolution.
    # Default is 100 (a bit blurry when zoomed).
    # 150 gives a sharper image suitable for
    # including in a report or resume portfolio,
    # without making the file unnecessarily huge.
    # ─────────────────────────────────────────
    plt.savefig(output_path, dpi=150)
    print(f"✅ Chart saved to: {output_path}")

    # ─────────────────────────────────────────
    # WHY plt.show()?
    # Opens the chart in a popup window so you can
    # see it immediately without opening the PNG
    # file manually. On some systems (servers,
    # certain terminals) this may not open a window —
    # the saved PNG is the reliable fallback.
    # ─────────────────────────────────────────
    plt.show()


# ─────────────────────────────────────────────
# WHY if __name__ == '__main__'?
# Same pattern as every other file — lets you
# run this standalone: python -m src.video.visualise
# It loads the timeline.json saved by process.py
# and generates the chart from it, WITHOUT
# re-running the entire video analysis.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TIMELINE_PATH = "data/samples/timeline.json"

    if not Path(TIMELINE_PATH).exists():
        print(f"❌ No timeline found at {TIMELINE_PATH}")
        print("   Run 'python -m src.video.process' first")
    else:
        timeline = load_timeline(TIMELINE_PATH)
        plot_timeline(timeline)
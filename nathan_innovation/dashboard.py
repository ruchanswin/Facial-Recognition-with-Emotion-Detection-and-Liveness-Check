import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nathan_innovation import stats
from nathan_innovation.db import get_events

def launch_dashboard() -> None:
    today = date.today().isoformat()

    attendees = stats.attendance_summary(today)
    emotions = stats.emotion_distribution(today)
    daily = stats.daily_counts(7)
    hourly = stats.arrivals_by_hour(today)
    avg_conf = stats.avg_confidence(today)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Attendance Dashboard — {today}   |   Present today: {len(attendees)}   |   Avg confidence: {avg_conf:.2f}",
        fontsize=13,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Top-left: today's attendee list ---
    ax_list = fig.add_subplot(gs[0, 0])
    ax_list.axis("off")
    ax_list.set_title("Today's Attendees", fontweight="bold")
    first_seen: dict[str, dict] = {}
    for ev in get_events(today):
        if ev["person_id"] not in first_seen:
            first_seen[ev["person_id"]] = ev
    if first_seen:
        lines = []
        for name in sorted(first_seen.keys()):
            ev = first_seen[name]
            time_str = datetime.fromisoformat(ev["timestamp"]).strftime("%H:%M")
            emotion = ev["emotion"] or "—"
            lines.append(f"{name:<18} {time_str}   {emotion}")
        ax_list.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax_list.transAxes,
            va="top", ha="left",
            fontsize=9,
            fontfamily="monospace",
        )
    else:
        ax_list.text(0.5, 0.5, "No attendees yet", transform=ax_list.transAxes,
                     va="center", ha="center", color="grey")

    # --- Top-right: emotion pie chart ---
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_pie.set_title("Emotion Distribution (Today)", fontweight="bold")
    if emotions:
        labels = list(emotions.keys())
        sizes = list(emotions.values())
        ax_pie.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    else:
        ax_pie.text(0.5, 0.5, "No data", transform=ax_pie.transAxes,
                    va="center", ha="center", color="grey")
        ax_pie.axis("off")

    # --- Bottom-left: daily attendance bar chart ---
    ax_daily = fig.add_subplot(gs[1, 0])
    ax_daily.set_title("Daily Unique Attendees (Last 7 Days)", fontweight="bold")
    if daily:
        days = [d[4:]  for d, _ in daily]  # strip year, keep MM-DD
        counts = [c for _, c in daily]
        bars = ax_daily.bar(days, counts, color="#4C72B0")
        ax_daily.set_ylabel("Unique attendees")
        ax_daily.set_ylim(0, max(counts) + 2 if counts else 5)
        for bar, val in zip(bars, counts):
            if val:
                ax_daily.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontsize=8,
                )
        ax_daily.tick_params(axis="x", labelrotation=30)

    # --- Bottom-right: arrivals by hour ---
    ax_hour = fig.add_subplot(gs[1, 1])
    ax_hour.set_title("Arrival Times (Today)", fontweight="bold")
    if hourly:
        hours = sorted(hourly.keys())
        counts_h = [hourly[h] for h in hours]
        ax_hour.bar([f"{h:02d}:00" for h in hours], counts_h, color="#55A868")
        ax_hour.set_ylabel("Arrivals")
        ax_hour.tick_params(axis="x", labelrotation=30)
    else:
        ax_hour.text(0.5, 0.5, "No data", transform=ax_hour.transAxes,
                     va="center", ha="center", color="grey")

    plt.show()


if __name__ == "__main__":
    launch_dashboard()

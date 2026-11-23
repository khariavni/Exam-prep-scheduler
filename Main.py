import json
import os
import sys
from copy import deepcopy
from datetime import date, datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
PLAN_FILE = os.path.join(BASE_DIR, "plan.json")
CSV_FILE = os.path.join(BASE_DIR, "plan.csv")
CHART_FILE = os.path.join(BASE_DIR, "plan_chart.png")
GRAY_FILE = os.path.join(BASE_DIR, "plan_chart_gray.png")
THUMB_FILE = os.path.join(BASE_DIR, "plan_chart_thumb.png")
DATE_FMT = "%Y-%m-%d"


def make_topics(prefix, count, weights=None):
    topics = []
    for i in range(1, count + 1):
        w = weights[i - 1] if weights else 3
        topics.append({"name": f"{prefix} {i}", "weight": w, "done": False})
    return topics


def pyq(n):
    return [{"name": f"PYQ practice {i}", "weight": 2, "done": False}
            for i in range(1, n + 1)]


DEFAULT_DATA = {
    "settings": {
        "plan_start": "2026-11-12",
        "base_session_hours": 3,
        "break_weekday": 6,
    },
    "subjects": [
        {"name": "Python", "exam_date": "2026-12-18", "difficulty": 3,
         "priority": True, "topics": make_topics("Unit", 5) + pyq(2)},
        {"name": "DSA", "exam_date": "2026-12-22", "difficulty": 5,
         "priority": True,
         "topics": make_topics("Unit", 6, [3, 3, 3, 5, 5, 5]) + pyq(2)},
        {"name": "OOPS", "exam_date": "2026-12-24", "difficulty": 3,
         "priority": True, "topics": make_topics("Unit", 5) + pyq(2)},
        {"name": "CLE", "exam_date": "2026-12-29", "difficulty": 2,
         "priority": False, "topics": make_topics("Unit", 4)},
        {"name": "MDS", "exam_date": "2026-12-31", "difficulty": 4,
         "priority": True, "topics": make_topics("Unit", 5) + pyq(2)},
    ],
}


def to_date(text):
    return datetime.strptime(text, DATE_FMT).date()


def pretty(d):
    return d.strftime("%a %d %b")


def days_between(a, b):
    return (b - a).days


class Sorter:
    name = "base"

    def __init__(self, key=lambda x: x, reverse=False):
        self.key = key
        self.reverse = reverse

    def sort(self, items, key=None, reverse=None):
        if key is not None:
            self.key = key
        if reverse is not None:
            self.reverse = reverse
        return self._sort(list(items))

    def _sort(self, items):
        raise NotImplementedError("each algorithm overrides _sort")


class MergeSort(Sorter):
    name = "merge"

    def _sort(self, items):
        if len(items) <= 1:
            return items
        mid = len(items) // 2
        left = self._sort(items[:mid])
        right = self._sort(items[mid:])
        merged, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            if self.reverse:
                take_left = self.key(left[i]) >= self.key(right[j])
            else:
                take_left = self.key(left[i]) <= self.key(right[j])
            if take_left:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged


class InsertionSort(Sorter):
    name = "insertion"

    def _sort(self, items):
        for i in range(1, len(items)):
            current = items[i]
            j = i - 1
            while j >= 0 and (self.key(items[j]) < self.key(current) if self.reverse
                              else self.key(items[j]) > self.key(current)):
                items[j + 1] = items[j]
                j -= 1
            items[j + 1] = current
        return items


class QuickSort(Sorter):
    name = "quick"

    def _sort(self, items):
        if len(items) <= 1:
            return items
        pivot = self.key(items[len(items) // 2])
        smaller = [x for x in items if self.key(x) < pivot]
        equal = [x for x in items if self.key(x) == pivot]
        bigger = [x for x in items if self.key(x) > pivot]
        if self.reverse:
            return self._sort(bigger) + equal + self._sort(smaller)
        return self._sort(smaller) + equal + self._sort(bigger)


SORTERS = {"merge": MergeSort, "insertion": InsertionSort, "quick": QuickSort}


class Topic:
    def __init__(self, name, weight=3, done=False):
        self.name = name
        self.weight = weight
        self.done = done

    def to_dict(self):
        return {"name": self.name, "weight": self.weight, "done": self.done}

    def __str__(self):
        return f"{self.name} (weight {self.weight})"


class Subject:
    def __init__(self, name, exam_date, difficulty, priority=False, topics=None):
        self.name = name
        self.exam_date = exam_date
        self.difficulty = difficulty
        self.priority = priority
        self.topics = topics if topics is not None else []
        self.queue = []
        self.revise = []
        self.rev_idx = 0

    @classmethod
    def from_dict(cls, d):
        topics = [Topic(t["name"], t["weight"], t["done"]) for t in d["topics"]]
        return cls(d["name"], d["exam_date"], d["difficulty"],
                   d.get("priority", False), topics)

    def to_dict(self):
        return {"name": self.name, "exam_date": self.exam_date,
                "difficulty": self.difficulty, "priority": self.priority,
                "topics": [t.to_dict() for t in self.topics]}

    def add_topic(self, name, weight=3):
        self.topics.append(Topic(name, weight))

    def exam(self):
        return to_date(self.exam_date)

    def pending(self):
        return [t for t in self.topics if not t.done]

    def score(self, day, left_names=None):
        if left_names is None:
            left_names = {t.name for t in self.pending()}
        days_left = max(days_between(day, self.exam()), 1)
        total_w = sum(t.weight for t in self.topics) or 1
        left_w = sum(t.weight for t in self.topics if t.name in left_names)
        return (self.difficulty * 2 + 40 / days_left
                + (2 if self.priority else 0) + 6 * left_w / total_w)

    def study_days_left(self, start, break_weekday):
        count, d = 0, start
        while d < self.exam():
            if d.weekday() != break_weekday:
                count += 1
            d += timedelta(days=1)
        return count

    def session_hours(self, settings):
        h = settings["base_session_hours"]
        if self.difficulty >= 4:
            h += 1
        elif self.difficulty <= 2:
            h -= 1
        return h


def load_data():
    if not os.path.exists(DATA_FILE):
        save_data(DEFAULT_DATA)
        print("(first run) created data.json with your default subjects")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_plan(plan):
    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)


class Planner:
    def __init__(self, data):
        self.data = data
        self.settings = data["settings"]
        self.subjects = [Subject.from_dict(s) for s in data["subjects"]]

    def rank(self, today, algo="merge"):
        active = [s for s in self.subjects if s.pending()]
        if not active:
            return []
        difficulty = np.array([s.difficulty for s in active])
        days_left = np.maximum(np.array([days_between(today, s.exam()) for s in active]), 1)
        focus = np.array([2 if s.priority else 0 for s in active])
        total_w = np.array([sum(t.weight for t in s.topics) or 1 for s in active])
        left_w = np.array([sum(t.weight for t in s.pending()) for s in active])
        scores = np.round(difficulty * 2 + 40 / days_left + focus + 6 * left_w / total_w, 2)
        scored = list(zip(scores.tolist(), active))
        return SORTERS[algo]().sort(scored, key=lambda pair: pair[0], reverse=True)

    def build_plan(self, today):
        settings = self.settings
        break_day = settings["break_weekday"]
        subjects = deepcopy(self.subjects)
        by_topic_weight = QuickSort(key=lambda p: (-p[1].weight, p[0]))

        for s in subjects:
            pending = [(i, t) for i, t in enumerate(s.topics) if not t.done]
            s.queue = [t.name for _, t in by_topic_weight.sort(pending)]
            every = by_topic_weight.sort(list(enumerate(s.topics)))
            s.revise = [t.name for _, t in every]
            s.rev_idx = 0

        by_exam = InsertionSort().sort(subjects, key=lambda s: s.exam())
        last_exam = by_exam[-1].exam()
        exams_on = {}
        for s in by_exam:
            exams_on.setdefault(s.exam_date, []).append(s.name)

        start = max(today + timedelta(days=1), to_date(settings["plan_start"]))
        top_first = MergeSort(key=lambda p: p[0], reverse=True)

        plan, d, last_subject = [], start, None
        while d <= last_exam:
            row = {"date": d.strftime(DATE_FMT), "day": d.strftime("%a"),
                   "type": "study", "subject": "", "topic": "", "hours": 0,
                   "exams": exams_on.get(d.strftime(DATE_FMT), [])}

            eve = [s for s in by_exam if not s.queue
                   and s.exam() == d + timedelta(days=1)]

            if d.weekday() == break_day:
                row["type"] = "break"
            elif eve:
                row.update(type="revision", subject=eve[0].name,
                           topic="Last-day revision (key units + PYQs)",
                           hours=eve[0].session_hours(settings))
                last_subject = eve[0].name
            else:
                candidates = [s for s in by_exam if s.queue and s.exam() > d]
                if not candidates:
                    upcoming = [s for s in by_exam if s.exam() > d]
                    if not upcoming:
                        row["type"] = "free"
                    else:
                        scored = [(s.score(d, set(s.queue))
                                   - (3 if s.name == last_subject else 0), s)
                                  for s in upcoming]
                        best = top_first.sort(scored)[0][1]
                        topic = best.revise[best.rev_idx % len(best.revise)]
                        best.rev_idx += 1
                        row.update(type="revision", subject=best.name,
                                   topic=f"Revise {topic}",
                                   hours=best.session_hours(settings))
                        last_subject = best.name
                else:
                    scored = []
                    for s in candidates:
                        score = s.score(d, set(s.queue))
                        if s.name == last_subject:
                            score -= 3
                        if s.study_days_left(d, break_day) <= len(s.queue):
                            score += 100
                        scored.append((score, s))
                    best = top_first.sort(scored)[0][1]
                    row["subject"] = best.name
                    row["topic"] = best.queue.pop(0)
                    row["hours"] = best.session_hours(settings)
                    last_subject = best.name
            plan.append(row)
            d += timedelta(days=1)
        return plan

    def mark_done(self, subject_name, topic_name):
        for s in self.data["subjects"]:
            if s["name"] == subject_name:
                for t in s["topics"]:
                    if t["name"] == topic_name:
                        t["done"] = True


def plan_frame(plan):
    df = pd.DataFrame(plan)
    df["date"] = pd.to_datetime(df["date"])
    df["exams"] = df["exams"].apply(", ".join)
    return df


def hours_summary(df):
    work = df[df["type"].isin(["study", "revision"])]
    table = work.pivot_table(index="subject", columns="type", values="hours",
                             aggfunc="sum", fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def export_csv(plan):
    plan_frame(plan).to_csv(CSV_FILE, index=False)
    return pd.read_csv(CSV_FILE, parse_dates=["date"])


def draw_chart(plan):
    df = plan_frame(plan)
    summary = hours_summary(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})
    ax1.bar(summary.index, summary["total"], color="#1B2F6B")
    ax1.set_title("Planned hours per subject")
    ax1.set_ylabel("hours")
    colours = {"Python": "#4A6FA5", "DSA": "#C0392B", "OOPS": "#E8A33D",
               "CLE": "#3C9D5D", "MDS": "#1B2F6B"}
    work = df[df["type"].isin(["study", "revision"])]
    for name, grp in work.groupby("subject"):
        ax2.bar(grp["date"], grp["hours"], label=name, color=colours.get(name, "gray"))
    ax2.set_title("Daily study hours")
    ax2.legend(fontsize=8, ncol=5, loc="upper center")
    ax2.set_ylim(0, 6.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=120)
    plt.close(fig)
    img = Image.open(CHART_FILE)
    gray = img.convert("L")
    gray.save(GRAY_FILE)
    thumb = gray.copy()
    thumb.thumbnail((400, 400))
    thumb.save(THUMB_FILE)
    return img.size, thumb.size


def print_timetable(plan):
    week_start = None
    for r in plan:
        d = to_date(r["date"])
        monday = d - timedelta(days=d.weekday())
        if monday != week_start:
            week_start = monday
            print(f"\n--- Week of {monday.strftime('%d %b')} " + "-" * 38)
        if r["type"] == "break":
            what = "Rest day"
        elif r["type"] == "free":
            what = "No study (exam day)" if r["exams"] else "Free day"
        else:
            what = f"{r['subject']:<7} {r['topic']}  ({r['hours']}h)"
        exam = f"   << EXAM: {', '.join(r['exams'])}" if r["exams"] else ""
        print(f"  {pretty(d)} | {what}{exam}")


def print_ranking(planner, today, algo):
    ranked = planner.rank(today, algo)
    print(f"\nPriority ranking for {pretty(today)}  (sorted with {algo} sort)")
    print(f"{'#':<3}{'Subject':<9}{'Exam':<12}{'Days left':<11}"
          f"{'Diff':<6}{'Topics left':<12}Score")
    for n, (score, s) in enumerate(ranked, 1):
        left = days_between(today, s.exam())
        print(f"{n:<3}{s.name:<9}{pretty(s.exam()):<12}"
              f"{left:<11}{s.difficulty:<6}{len(s.pending()):<12}{score}")


def print_workload(planner, today):
    brk = planner.settings["break_weekday"]
    start = max(today + timedelta(days=1), to_date(planner.settings["plan_start"]))
    print("\nWorkload check (study days before each exam vs topics left)")
    for s in InsertionSort().sort(planner.subjects, key=lambda s: s.exam()):
        days = s.study_days_left(start, brk)
        left = len(s.pending())
        flag = "OK" if days >= left else "!! not enough days"
        print(f"  {s.name:<7} {left:>2} topics | {days:>2} study days | {flag}")


def ask(prompt, default=None):
    text = input(prompt).strip()
    return text if text else default


def ask_int(prompt, low, high):
    while True:
        text = input(prompt).strip()
        if text.isdigit() and low <= int(text) <= high:
            return int(text)
        print(f"  enter a number from {low} to {high}")


def ask_date(prompt):
    while True:
        text = input(prompt).strip()
        try:
            to_date(text)
            return text
        except ValueError:
            print("  use format YYYY-MM-DD, e.g. 2026-12-18")


def action_show_plan(planner, today):
    plan = planner.build_plan(today)
    save_plan(plan)
    print_timetable(plan)
    print("\n(saved to plan.json)")


def action_today(planner, today):
    plan = planner.build_plan(today)
    print(f"\nToday is {pretty(today)}. Next 3 days:")
    for r in plan[:3]:
        d = to_date(r["date"])
        if r["type"] in ("study", "revision"):
            print(f"  {pretty(d)}: {r['subject']} - {r['topic']} ({r['hours']}h)")
        else:
plan)
    print(f"\nSaved plan_chart.png {big[0]}x{big[1]}, "
          f"plan_chart_gray.png (print version) and plan_chart_thumb.png {small[0]}x{small[1]}")


def action_table(planner, today):
    plan = planner.build_plan(today)
    df = export_csv(plan)
    print(f"\nSaved plan.csv ({len(df)} rows) and read it back with pandas.")
    print("\nHours per subject:")
    print(hours_summary(df).to_string())
    print("\nFirst 3 rows (iloc):")
    print(df.iloc[:3][["date", "subject", "topic", "hours"]].to_string(index=False))
    busiest = df.loc[df["hours"].idxmax()]
    print(f"\nLongest day (loc): {pretty(busiest['date'].date())} - "
          f"{busiest['subject']} ({busiest['hours']}h)")


MENU = """
==========  EXAM PREP SCHEDULER  ==========
 1. Show full timetable
 2. What do I study today / next 3 days
 3. Mark a session as done
 4. Priority ranking (pick sorting algorithm)
 5. Workload check
 6. Add a subject
 7. Edit a subject (difficulty / exam date)
 8. Charts (Matplotlib + image export)
 9. Plan table (pandas, saves plan.csv)
 0. Exit
"""


def main():
    today = date.today()
    if "--today" in sys.argv:
        today = to_date(sys.argv[sys.argv.index("--today") + 1])
    if "--test" in sys.argv:
        self_test()
        return

    planner = Planner(load_data())
    print(f"H

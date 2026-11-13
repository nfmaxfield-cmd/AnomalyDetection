# Questions to Nick's side

Use one file per question, named `YYYY-MM-DD-short-topic.md` (for example `2026-11-20-camera-check-fails.md`).
Commit and push it, or send it to Nick. The answer is written into the same file, under **Answer**.

Copy this into the new file:

```
# <one-line question>

Type: bug | question | feature request | change made
Asked by: Josh / Chris / their AI, <date>
Status: open

## What we did
<the exact commands, in order, and which window>

## What we saw
<the full output, pasted as text; the runs/<date-time>/ folder name; which picture, if any (send pictures
another way, never in git)>

## What we expected

## Setup
<computer and operating system, Python version (python --version), webcam model, arm, git log --oneline -1>

## Answer
(Nick's side writes here)
```

**Type** says what kind of entry it is:
- **bug:** the software did something wrong.
- **question:** how to do something, or what something means.
- **feature request:** the software should do something new.
- **change made:** you changed the software yourselves. Say what changed and why.

**Status** is `open` until it's answered, then `answered`, and `closed` once you've confirmed it.

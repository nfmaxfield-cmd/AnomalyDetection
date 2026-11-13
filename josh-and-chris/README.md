# Josh and Chris's folder

This folder is for you and your AI assistant: Claude, ChatGPT, Copilot or any other.

**Who does what:** Nick is responsible for the **software** in this repository. Everything else is up to
you: the station's hardware, the arm and its program, the parts, collecting pictures, the business numbers,
testing on the real station, and your course work.

## What is in here

| File | What it is | Who writes it |
|---|---|---|
| `README.md` | this page | Nick |
| `AGENTS.md` | instructions for your AI: what this project is, what it may and may not do, how to ask Nick's side a question | Nick |
| `CLAUDE.md` | the same instructions, in the file name Claude Code looks for | Nick |
| `SYSTEM-OUTLINE.md` | the detailed outline of how the whole station gets built, in phases, with who owns each part | Nick (you may add notes) |
| `questions/` | questions from you or your AI to Nick's side, and the answers, one file per question | both |
| `notes/` | your own notes, decisions and progress (`notes/status.md` to start) | you |

## How to use it with your AI

1. Open your AI in the AnomalyDetection folder. If it can read files (Claude Code, Cursor, Copilot in VS
   Code), it reads `AGENTS.md` or `CLAUDE.md` by itself. For a chat AI with no file access, paste in
   `AGENTS.md` and `SYSTEM-OUTLINE.md` at the start.
2. Ask it what you need ("help me build phase 2", "why did this part get rejected?").
3. When something needs Nick's side (a bug, a missing feature, a result that looks wrong), ask your AI to
   write a question file in `questions/` (the format is in `questions/README.md`). Then commit and push it,
   or send it to Nick. The answer is written into the same file.

## Rules for this repository

- **It is public.** Never put passwords, keys or personal details anywhere in it.
- **Pictures and models stay out of git.** The `data/`, `models/` and `runs/` folders are ignored. Send
  pictures another way (shared drive, USB stick).
- **This folder is yours.** Change anything in it.
- **Software changes.** Nick doesn't mind you changing the software without asking first. Two things help
  keep it reliable:
  - tell Nick's side in a question file what you changed and why;
  - run the tests (`python -m pytest`) before you push.

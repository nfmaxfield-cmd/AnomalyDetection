#!/usr/bin/env python3
"""Example for the arm's code: how to send "empty" and "paused" and act on the reply.

Changelog
  0.1.0  2026-11-12  first version.

Copy the three small functions into the arm program. Start the station first, in its own window:
    python -m qc run
Then this example (in a second window) walks one part through both faces:
    python tools/arm_client_example.py --part-id p1 --shape square
The move_* functions are placeholders for the arm's own commands.
"""
import argparse
import json
import urllib.request

STATION = "http://127.0.0.1:8765"


def ask(cmd, **body):
    """Send one request to the station and return its reply (a dict)."""
    req = urllib.request.Request(STATION + "/" + cmd, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def move_away_from_spot():
    print("  (arm) moving clear of the spot")


def flip_part():
    print("  (arm) flipping the part")


def place(pile):
    print("  (arm) placing the part on the %s pile" % pile.upper())


def inspect_part(part_id, shape):
    """Top face, then (unless already rejected) the bottom face. Returns the pile."""
    rep = ask("paused", part_id=part_id, shape=shape, face="top")
    print("  top:", rep.get("decision"), rep.get("reason") or "", rep.get("warnings") or "")
    if rep.get("decision") != "reject":
        flip_part()
        rep = ask("paused", part_id=part_id, shape=shape, face="bottom")
        print("  bottom:", rep.get("decision"), rep.get("reason") or "", rep.get("warnings") or "")
    pile = "reject" if rep.get("decision") == "reject" or rep.get("part_decision") == "reject" else "accept"
    place(pile)
    if rep.get("empty_due"):                      # the station asks for a fresh empty picture now and then
        move_away_from_spot()
        print("  empty:", ask("empty"))
    return pile


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-id", default="p1")
    ap.add_argument("--shape", default="square")
    ap.add_argument("--first", action="store_true", help="send 'empty' first (start of a session)")
    a = ap.parse_args()
    if a.first:
        move_away_from_spot()
        print("empty:", ask("empty"))
    print("part %s -> %s" % (a.part_id, inspect_part(a.part_id, a.shape)))

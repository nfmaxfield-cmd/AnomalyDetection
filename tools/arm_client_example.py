#!/usr/bin/env python3
"""Example for the arm's code: how to send "empty" and "paused" and act on the reply.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes: reads needs_attention and recheck (a part the station could not judge goes
                     to reject, never accept; "recheck" asks again up to RECHECKS times); accept only when
                     part_decision is accept; a refused "empty" is retried after the spot is cleared.

Copy the small functions (ask, ask_face, inspect_part, send_empty) into the arm program. Start the station first, in its own window:
    python -m qc run
Then this example (in a second window) walks one part through both faces:
    python tools/arm_client_example.py --part-id p1 --shape square
The move_* functions are placeholders for the arm's own commands.
"""
import argparse
import json
import time
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


RECHECKS = 2        # [chosen] how many times to ask again when the station says "recheck"; not DOE (a cycle-time choice)


def ask_face(part_id, shape, face):
    """One face: ask, and when the station says "recheck" (e.g. the part was still moving) wait a moment and
    ask again for the same part and face. The last answer counts."""
    rep = ask("paused", part_id=part_id, shape=shape, face=face)
    tries = 0
    while rep.get("recheck") and tries < RECHECKS:
        tries += 1
        print("  %s: asked to look again (%s)" % (face, rep.get("reason")))
        time.sleep(0.5)
        rep = ask("paused", part_id=part_id, shape=shape, face=face)
    print("  %s:" % face, rep.get("decision"), rep.get("reason") or "", rep.get("warnings") or "")
    if rep.get("needs_attention"):
        # the station could not judge this picture properly: the part goes to REJECT (never accept), and a
        # person should look at the reason (light changed, something on the backdrop, wrong face up ...)
        print("  ATTENTION: %s" % rep.get("reason"))
    return rep


def inspect_part(part_id, shape):
    """Top face, then (unless already rejected) the bottom face. Returns the pile."""
    rep = ask_face(part_id, shape, "top")
    if rep.get("decision") != "reject":
        flip_part()
        rep = ask_face(part_id, shape, "bottom")
    # "accept" only when the station said accept for the whole part; anything else goes to reject
    pile = "accept" if rep.get("decision") == "accept" and rep.get("part_decision") == "accept" else "reject"
    place(pile)
    if rep.get("empty_due"):                      # the station asks for a fresh empty picture now and then
        move_away_from_spot()
        send_empty()
    return pile


def send_empty():
    """Send "empty". If the station refuses it (something is still in view), clear the spot and try again."""
    while True:
        r = ask("empty")
        print("  empty:", "ok" if r.get("ok") else r.get("reason"))
        if r.get("ok"):
            return r
        input("  Clear the spot, then press Enter to send \"empty\" again ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-id", default="p1")
    ap.add_argument("--shape", default="square")
    ap.add_argument("--first", action="store_true", help="send 'empty' first (start of a session)")
    a = ap.parse_args()
    if a.first:
        move_away_from_spot()
        send_empty()
    print("part %s -> %s" % (a.part_id, inspect_part(a.part_id, a.shape)))

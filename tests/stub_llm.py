"""A canned LLM so the pipeline can be exercised end-to-end without an API key.

Returns realistic, correctly-shaped responses for every stage. Used by tests/run_offline.py to
prove the integration works and to produce a sample video.
"""
from __future__ import annotations

import random
import re


class StubLLM:
    def __init__(self, seed: int = 0):
        self.calls = 0
        self.rng = random.Random(seed)

    def complete(self, prompt: str, **kw) -> str:
        raise NotImplementedError

    def complete_json(self, prompt: str, **kw):
        self.calls += 1
        # Dispatch on markers unique to each prompt file's JSON output block. Order matters:
        # several prompts mention "hook" and "punch" in prose, so match on schema keys only.
        if "Candidate A" in prompt and "Candidate B" in prompt:
            return self._judge()
        if "beats_baseline" in prompt:
            return self._qc()
        if "pause_before_ms" in prompt:
            return self._direction(prompt)
        if "character_sheet" in prompt:
            return self._shots(prompt)
        if "punchline_mechanism" in prompt or "alternative final lines" in prompt:
            return self._punchup()
        if "the_joke" in prompt:
            return self._script()
        if '"premises"' in prompt:
            return self._premises()
        if "description_hook" in prompt:
            return self._metadata()
        raise AssertionError(f"stub has no branch for this prompt: {prompt[:200]!r}")

    def _judge(self):
        return {"winner": self.rng.choice(["A", "B"]),
                "deciding_criterion": "specificity",
                "why": "one candidate carried a detail the other only gestured at",
                "loser_flaw": "too general", "confidence": "medium"}

    def _qc(self):
        return {"beats_baseline": True, "deciding_criterion": "specificity",
                "why": "the candidate commits to one concrete object and stays with it",
                "vetoes": {"explains_its_own_joke": False, "punch_is_not_last": False,
                           "ai_tell_cluster": False, "generic_premise": False,
                           "no_recognisable_truth": False, "unsafe": False},
                "weakest_line_index": 2, "one_fix": "tighten the third line"}

    def _premises(self):
        base = [
            ("A colleague labels every item in the office fridge with a date on masking tape",
             "He starts dating things that cannot spoil, including the fridge itself",
             "masking tape", "bad system", "stranger"),
            ("A man refuses to use the office lift and takes the stairs to the eleventh floor",
             "He arrives at every meeting unable to speak for the first four minutes",
             "eleventh floor", "escalating commitment", "friend"),
            ("Someone brings a full mechanical keyboard to a coffee shop",
             "He apologises for the noise while continuing to type louder",
             "mechanical keyboard", "misplaced confidence", "stranger"),
            ("A team keeps a shared spreadsheet of whose turn it is to reply to a client",
             "The spreadsheet now has a change log and nobody has replied to the client",
             "change log", "bad system", "institution"),
            ("A man microwaves his lunch in ninety second bursts and stands guard",
             "He explains the burst schedule to anyone who walks past the kitchen",
             "ninety second bursts", "sincere wrong effort", "friend"),
            ("Someone prints an email to read it, then annotates it by hand",
             "He then types the annotations back into a reply and prints that too",
             "annotations", "unspoken rule", "stranger"),
        ]
        return {"premises": [
            {"id": i + 1, "situation": s, "turn": t, "detail": d, "mechanism": m,
             "target": tg, "has_screen": False, "has_other_people": True}
            for i, (s, t, d, m, tg) in enumerate(base)]}

    def _script(self):
        """A realistic ~105-word script, so the offline run exercises a video that actually
        clears the gate's 20s floor rather than being rejected for length."""
        return {"beats": [
            {"role": "hook", "text": "There is a man in my office who puts dates on the fridge items."},
            {"role": "setup", "text": "Not his name on the tape. The date. Every single container, in the same handwriting."},
            {"role": "escalate", "text": "Last Tuesday I found a date on a jar of mustard that somebody had opened in March."},
            {"role": "escalate", "text": "Mustard does not expire. Mustard outlives buildings. Mustard will be here after all of us."},
            {"role": "escalate", "text": "I asked him about it and he took out his phone and showed me a photograph of the shelf."},
            {"role": "turn", "text": "He said, and I am quoting him exactly here, we needed a baseline."},
            {"role": "punch", "text": "There is now a date on the fridge."},
        ], "word_count": 106,
            "the_joke": "a person applying a system with total sincerity to something that does not need one"}

    def _punchup(self):
        return {"candidates": [
            {"id": 1, "text": "There is now a date on the fridge itself.", "mechanism": "escalation", "risk": "safe"},
            {"id": 2, "text": "I checked this morning. The tape has a tape.", "mechanism": "escalation", "risk": "risky"},
            {"id": 3, "text": "He has never once eaten anything from that fridge.", "mechanism": "reversal", "risk": "risky"},
            {"id": 4, "text": "The mustard is fine. Everyone else is not.", "mechanism": "deflation", "risk": "safe"},
        ]}

    def _lines_from(self, prompt):
        return re.findall(r"^\s*(\d+)\.\s*\[(\w+)\]\s*(.+)$", prompt, re.MULTILINE)

    def _direction(self, prompt):
        pauses = {"hook": 0, "setup": 130, "escalate": 110, "turn": 280, "punch": 540, "tag": 320}
        out = []
        for idx, role, text in self._lines_from(prompt):
            i = int(idx)
            out.append({"index": i, "role": role, "text": text.strip(),
                        "direction": "deadpan" if role == "punch" else None,
                        "pause_before_ms": 0 if i == 0 else pauses.get(role, 120),
                        "emphasis": [max(text.split(), key=len).strip(".,").upper()]})
        return {"lines": out, "voice_note": "flat, unbothered, slightly tired"}

    def _shots(self, prompt):
        sizes = ["wide", "close", "medium", "extreme-close", "over-shoulder", "medium"]
        moves = ["push-in", "drift-left", "pull-out", "static-float", "drift-right", "push-in"]
        desc = [
            "wide shot of a man standing in front of an open office fridge, flat stare, harsh overhead strip light",
            "extreme close up of a strip of masking tape on a plastic container, hard side light",
            "medium shot of a man holding a jar up to the light, studying it, unimpressed expression",
            "close up of a jar of mustard alone on an empty shelf, single overhead light",
            "medium shot of a man speaking earnestly to an unseen colleague, hands apart explaining",
            "wide shot of a closed office fridge alone in an empty kitchen, one strip of tape on the door",
        ]
        return {"character_sheet": "a tired man in his thirties, short dark hair, grey zip fleece",
                "shots": [{"line_index": int(idx), "shot_size": sizes[i % len(sizes)],
                           "prompt": desc[i % len(desc)], "motion": moves[i % len(moves)],
                           "why_this_image": "carries the beat"}
                          for i, (idx, role, _t) in enumerate(self._lines_from(prompt))]}

    def _metadata(self):
        return {"title": "my colleague put a date on the mustard #shorts",
                "description_hook": "He has never eaten anything from that fridge. Not once.",
                "hashtags": "#shorts #comedy #officehumor #worklife #relatable",
                "tags": ["comedy", "shorts", "office humor", "work", "relatable", "standup"]}

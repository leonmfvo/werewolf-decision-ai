"""
Role composition configuration.

Changing the game's role composition (adding a Seer, a Madman,
changing headcounts, etc.) should ideally only require editing
THIS file -- main.py and the AI modules read from here instead of
hard-coding role names, so they can generalize across different
compositions without being rewritten each time.

Currently only the CURRENT 9-player composition is fully wired up
end-to-end (Hunter's guard ability, Medium's report ability, Seer's
investigate ability, etc.
are still role-SPECIFIC code in main.py -- adding a genuinely new
role with its own night ability still requires writing a new
<role>_ai.py module and hooking its ability into main.py, the same
way hunter_ai.py / medium_ai.py were built). What IS now general,
and will automatically adapt to ANY role list defined here, is:
  - the CO/claim mechanic (who can claim what, wolves' fake-claim
    options, contested/sole/provably-fake detection)
  - village_ai's execution-scoring features
"""

ROLES = ["Wo", "Wo", "Ps", "Hn", "Sy", "Kg", "Vi", "Vi", "Vi"]
# This module-level list is now only the FALLBACK/default 9-player
# composition (kept so any code that imports config.ROLES before a
# game starts still sees a sane 9-length list). The actual per-game
# role list main.py uses is built fresh each game by draw_roles()
# below -- see ROLE_POOL_10's comment for why.
#
# 9-player composition (up from the original 7): the Seer ("Sy",
# 占い師) was added on request, alongside one extra plain Villager so
# every EXISTING role keeps its original headcount rather than one
# being displaced to make room. Wolves are deliberately kept at 2
# (not scaled up to 3) -- 2/9 (~22%) is still within the normal human
# Werewolf convention for this player count, and changing the wolf
# ratio is a separate, hard-to-reverse balance decision the user
# didn't ask for; this can be revisited later with real measurement
# if 2 wolves turns out to be too weak/strong at 9 players.
#
# STEP 1 of the Madman ("Kg", 狂人) addition: one plain Villager is
# replaced by a Madman (headcount stays at 9). The Madman:
#   - WINS WITH THE WEREWOLVES (self_play.py's SIDE_INFO["madman"]
#     uses the "Werewolf" winner label), but
#   - is COUNTED AS VILLAGE for the numeric win-condition check
#     (main.py's check_game_end already does this for free -- it
#     only ever subtracts literal "Wo" roles, so a "Kg" naturally
#     falls into the "village" headcount bucket, exactly matching
#     standard Werewolf/Jinro rules where the Madman pads the human
#     headcount while secretly rooting for the wolves), and
#   - reads as INNOCENT to every investigative ability (Seer, Medium)
#     -- also already free, since both of those only ever check
#     `ROLES[x] == "Wo"`, never "is this player on the wolf side".
# STEP 2 converted the claim-certainty hard rules (village_ai.py's
# former absolute_fakes/absolute_wolves auto-execute overrides, and
# main.py's "Seer cascade") into LEARNED probabilities, since "only
# wolves ever fake-claim a role" stopped being safely assumable the
# moment a Madman could bluff too -- done BEFORE Step 3 below could
# safely go in, on purpose (see village_ai.choose_execution_target's
# docstring for the full reasoning).
#
# STEP 3 gives the Madman its fake-claim ability: it can claim
# Ps/Hn/Sy from Day 1 onward (main.py's madman_day_co) and produce the
# same fabricated daily reports a fake Medium/Seer claimant needs
# (madman_ai.choose_fake_report_strategy / choose_fake_seer_report_
# strategy), reusing wolf_ai's public-information-only CO/fake-report
# machinery wherever the Madman's lack of a known teammate doesn't
# change anything (see madman_ai.py's module docstring for exactly
# where it does).
#
# STEPS 4-5 (this update) complete the plan: once only 3 or 4 players
# remain, the Madman can openly reveal itself hostile to the village
# (main.py's madman_endgame_reveal / madman_ai's "STEP 4" section),
# using a hypergeometric estimate of how many wolves are still alive
# from public information only (village_ai.estimate_wolf_alive_
# distribution) to judge whether the reveal is worth it. A genuine
# Villager can, in that SAME narrow endgame window, falsely declare
# the same hostility (main.py's villager_endgame_bluff / villager_ai's
# "ENDGAME BLUFF" section) -- both write into the exact same public
# main.py list (self_declared_non_village), which is what keeps a
# real reveal from being an automatic, unambiguous tell.
#
# This completes the originally-planned 5-step Madman addition.
CLAIMABLE_ROLES = ["Ps", "Hn", "Sy"]

# Roles with a genuine, unique holder that's meaningful to CLAIM
# publicly (i.e. roles with SOME kind of information/ability worth
# announcing). "Wo" and "Vi" are deliberately excluded here -- a
# wolf's "claim to be a villager" and a real villager both fall
# under the separate is_villager_claimant / claim_villager
# mechanic, not this list.

# Display-only Japanese labels.
ROLE_JP = {"Wo": "狼", "Ps": "霊", "Hn": "狩", "Sy": "占", "Kg": "狂", "Vi": "村"}

# ============================================================
# "10a狩" composition (2026-10-03 addition, user's naming)
# ============================================================
#
# User's framing: imagine a 10-player pool where the wolves get a
# free kill before the game is even observed, removing one non-wolf
# player entirely. Mechanically this does NOT need an extra player
# slot, a Night-0 phase, or any board-state change at all -- it
# collapses to a much simpler rule the user confirmed explicitly:
# draw this game's 9-player ROLES from a 10-role pool by dropping
# ONE of the 8 non-wolf slots uniformly at random. Four of those 8
# slots are already-interchangeable plain Villagers, so HALF the
# time this drops a redundant "Vi" and the result is byte-for-byte
# today's existing fixed 9-player composition; the other half, one
# of Kg/Sy/Ps/Hn is dropped and that role is simply ABSENT from the
# game from the very start (not silent, not dead -- it never existed
# this game), with a 4th Vi filling the vacated seat.
#
# This is the "役欠け" (role-absence) feature: distinct from a role
# existing but staying silent (already handled everywhere via board/
# co_history state) or dying later (ditto) -- here the role's real
# holder never exists at all. Checked 2026-10-03: nothing in main.py
# or the AI modules does an unguarded ROLES.index("Sy")-style lookup
# that would crash on absence; every lookup already goes through
# next((...), None) or enumerate(ROLES)-based scans that degrade to
# "no such player" gracefully (see e.g. main.py's day-0 free Seer
# investigation). The one confirmed-unsafe numeric assumption was
# wolf_ai.py's alive_count normalization (see ALIVE_COUNT_NORM
# there), now fixed to read off this pool's size instead of a bare
# "9.0" literal.
ROLE_POOL_10 = ["Wo", "Wo", "Kg", "Sy", "Ps", "Hn", "Vi", "Vi", "Vi", "Vi"]

# Every game always ends up with exactly this many seated players
# (one non-wolf pool slot is always dropped, never a wolf slot) --
# this was already implicitly assumed as a bare "9.0" literal in
# wolf_ai.py's accuse-target feature normalization; named here so it
# no longer has to be hunted down as a magic number if the pool size
# ever changes again.
TOTAL_PLAYERS = len(ROLE_POOL_10) - 1


# ------------------------------------------------------------
# Mode switch (2026-10-06): "absence" (default, the 10-pool-minus-one
# draw above) or "9std" (the fixed standard 9: Wo Wo Ps Hn Sy Kg Vi Vi Vi,
# no role is ever absent -> fewer board buckets). Set with the
# environment variable WEREWOLF_MODE=9std, or generations.py --mode 9std.
# Each mode keeps its OWN weights file / generations folder so the two
# lines of training never overwrite each other.
# ------------------------------------------------------------
import os as _os

MODE = _os.environ.get("WEREWOLF_MODE", "absence")
ROLE_ABSENCE = MODE != "9std"
TAG = _os.environ.get("WEREWOLF_TAG", "")  # optional run name, e.g. "lr003"
_sfx = ("_" + TAG) if TAG else ""
WEIGHTS_FILE = ("trained_weights" if ROLE_ABSENCE else "trained_weights_9std") + _sfx + ".pkl"
GEN_DIR_NAME = ("generations" if ROLE_ABSENCE else "generations_9std") + _sfx


def draw_roles():
    """
    Build one game's actual 9-player ROLES list by dropping one
    uniformly-random non-wolf slot from ROLE_POOL_10 (see that
    constant's comment for the full reasoning). Call this once per
    game, from main.reset_game() -- NOT at import time -- so every
    self-play game gets an independent draw.
    """
    import random
    pool = list(ROLE_POOL_10)
    if not ROLE_ABSENCE:
        # 9-standard: always drop a plain Villager (never an ability role).
        drop = random.choice([i for i, role in enumerate(pool) if role == "Vi"])
        del pool[drop]
        random.shuffle(pool)
        return pool
    non_wolf_indices = [i for i, role in enumerate(pool) if role != "Wo"]
    drop = random.choice(non_wolf_indices)
    del pool[drop]
    random.shuffle(pool)
    return pool

import random

import config
import wolf_ai
import village_ai
import medium_ai
import hunter_ai
import seer_ai
import villager_ai
import madman_ai
import board_split


# ============================================================
# Game Initialization
# ============================================================

ROLES = config.ROLES
CLAIMABLE_ROLES = config.CLAIMABLE_ROLES

PLAYERS = [f"P{i}" for i in range(1, len(ROLES) + 1)]

# 表示用の日本語変換(実際の内部処理は英字コードのまま扱う)
ROLE_JP = config.ROLE_JP
CO_JP = {role: f"{ROLE_JP[role]}CO" for role in CLAIMABLE_ROLES}
CO_JP["Vi"] = "村CO"
CO_JP[None] = "－"


_DAY1_CO_DISPLAY_ORDER = ("Sy", "Ps", "Hn")  # 占->霊->狩, per the user's
                                              # requested reading order
                                              # (2026-10-02: changed from
                                              # the original 霊->占->狩)
                                              # (2026-09-30) -- purely a
                                              # DISPLAY grouping, unrelated
                                              # to the actual decision
                                              # order (medium->hunter->
                                              # seer->wolves->madman) that
                                              # day_phase(1) still runs in.
_day1_co_display_buffer = []  # [(claimant_idx, role_code), ...], Day-1 only
_missed_counted_days = set()  # execution days already added to missed_wolf_vote_count
_day1_flushed = False  # True once the Day-1 CO block has been printed


def announce_co(claimant_idx, role_code, verbose=True, day=None):
    """
    Unified CO announcement. On any day other than Day 1, this still
    prints the instant a claim happens (time-ordered), same as always.

    On Day 1 specifically (day=1 passed by the caller), the actual
    claim order is medium -> hunter -> seer -> wolves -> madman (see
    day_phase's own comments for why), which mixes real and fake
    claims of the same role together in a way that's a bit annoying to
    read (e.g. real-Ps, real-Hn, real-Sy, fake-Ps, fake-Sy). Per the
    user's explicit request, Day-1 announcements are instead BUFFERED
    here and printed all together, grouped by role (占/Sy -> 霊/Ps ->
    狩/Hn), via flush_day1_co_announcements() -- called once, right
    after every Day-1 CO decision (including the wolves'/Madman's
    fake claims) has been made. This changes DISPLAY ORDER ONLY; the
    underlying decision order, and every other day's announcements,
    are completely unaffected.
    """
    if day == 1 and not _day1_flushed:
        _day1_co_display_buffer.append((claimant_idx, role_code))
        return
    if verbose:
        print(f"{PLAYERS[claimant_idx]}【{CO_JP[role_code]}】")


def flush_day1_co_announcements(verbose=True):
    """
    Prints every Day-1 CO claim buffered by announce_co(..., day=1),
    grouped by role in _DAY1_CO_DISPLAY_ORDER (占->霊->狩), each group
    in seat-number order (2026-10-06: it used to be claim order, which always put the real claimant first). Then appends the
    "盤面: 占-霊-狩人数" board-shape notation (matching the "占-霊-狩"
    convention train_and_report.py's day1_config_report already uses),
    computed via _current_day1_config() -- safe here because this is
    called only after every Day-1 CO decision has already been made
    (see that function's own docstring for why it's UNSAFE mid-phase).
    Clears the buffer afterward so a later game doesn't leak into it.
    """
    global _day1_co_display_buffer, _day1_flushed
    _day1_flushed = True

    if verbose:
        order_index = {role: i for i, role in enumerate(_DAY1_CO_DISPLAY_ORDER)}
        for claimant_idx, role_code in sorted(
            _day1_co_display_buffer, key=lambda pair: (order_index.get(pair[1], len(order_index)), pair[0])
        ):
            print(f"{PLAYERS[claimant_idx]}【{CO_JP[role_code]}】")

        print(f"盤面(占-霊-狩): "
              f"{board_split.format_day1_config_detailed(_day1_co_display_buffer, ROLES)}")

    _day1_co_display_buffer = []

board = None
co_history = None
claim_day = None  # which day each player made their CO claim (None if never)
history = None  # decision log for the current game (state/action pairs)
guard_target = None  # who the hunter is protecting THIS night (private)
last_guard_choice = None  # the hunter's guard target from the PREVIOUS night (private)
last_guard_saved = False  # did the hunter's guard block last night's attack?
guard_ever_succeeded = False  # has a guard EVER succeeded so far this game? (public)
guard_success_count = 0  # HOW MANY TIMES has a guard succeeded so far? (public)
guard_success_targets = []  # who was saved -- these players are CONFIRMED non-wolf
                             # (wolves never attack their own), revealed once the
                             # real Hunter COs.
last_night_victim = None  # who died to last night's attack (public, None if peaceful)
attack_victims = []  # every player who has ever actually DIED to a night attack
                      # (not execution) -- publicly, unambiguously observable (a
                      # named player is simply dead the next morning, with no
                      # claimant's self-report standing between that fact and
                      # everyone seeing it), and just as certain a "not a wolf"
                      # proof as guard_success_targets: wolves never attack their
                      # own. Unlike guard_success_targets, this needs no Hunter-
                      # identity gating at all -- there's no ambiguous claimant
                      # whose account has to be trusted first, the death itself
                      # IS the public information.
last_execution_target = None  # who was executed YESTERDAY (for the medium's report)
execution_history = None  # list of every player index executed so far this game
_label_history = []  # one entry per _describe_execution_choice call (final label fixed up retroactively)
_pending_roll = None  # see _describe_execution_choice's docstring -- tracks an
                       # unresolved multi-claimant same-role contest left open
                       # by YESTERDAY's execution, across to TODAY's, so today's
                       # label can say whether the roll was completed or
                       # abandoned. {"role": claimed_role_code, "others": [idx,...]}
                       # (the OTHER alive claimants of that role, besides
                       # yesterday's executed target, at the moment of
                       # yesterday's decision) or None (no open roll pending).
medium_reports = None  # public list of {"claimant","day","player","is_wolf"} --
                        # ONLY ever populated with the REAL Medium's own true
                        # reports (never a fake claimant's -- see
                        # fake_medium_reports for those), and only once each
                        # day's report becomes legitimately trustworthy (see
                        # medium_report's docstring/comments for the backfill
                        # logic that recovers days lost while a Day-1 contest
                        # was still unresolved).
medium_claims = None  # public, ATTRIBUTED list of EVERY Medium claimant's spoken verdicts -- real or fake alike, same shape (2026-10-05 asymmetry fix)
fake_medium_reports = None  # public list of {"claimant","day","player","is_wolf"} --
                             # daily FAKE verdicts from wolf Medium claimants, which
                             # may or may not match the real medium_reports.
guard_instructions = None  # {claimant_idx: assigned_target_idx}, set when 2 Hunter
                            # claimants exist -- each is publicly told a DIFFERENT
                            # target to guard, so whichever one's assigned target
                            # dies is proven fake (only the real Hunter's guard
                            # actually blocks an attack).
confirmed_fake_hunters = None  # public list of claimant indices proven fake this way
confirmed_real_hunter = None  # the claimant index proven genuine this way, if any
fake_declarations = None  # {claimant_idx: [target_idx, ...]} -- a FAKE Hunter claim's lies, one per peaceful night
                           # about who they "guarded" (see wolf_ai's
                           # choose_fake_declaration_strategy). Meaningless on its
                           # own until guard-instruction verification resolves
                           # who's genuine -- see build_public_state.
nominations = None  # {claimant_idx: target_idx} -- each Medium claimant
                     # (real OR fake) publicly names one gray player as a
                     # suspect ("灰指定") right after claiming. If that
                     # happens to be the hidden Hunter, it's part of what
                     # can trigger their emergency-reveal decision. Kept
                     # per-claimant (not a single value) so the village can
                     # learn to weigh a SOLE claimant's nomination
                     # differently from a CONTESTED claimant's.
emergency_reveals = None  # public list of player indices whose Hunter claim
                           # was specifically triggered by the emergency
                           # (about-to-be-executed) path in progression_ai,
                           # as opposed to their own routine daily choice.
                           # Wolves can react to this -- but only from the
                           # NEXT day onward, since it happens AFTER wolf_day_co
                           # runs each day (see day_phase's call order).
emergency_wolf_claims = None  # public list of player indices whose "Ps" OR
                               # "Sy" claim was specifically made through the
                               # SAME emergency-evasion path (progression_ai),
                               # i.e. only AFTER that day's CO phase AND
                               # nomination had already happened. Unlike a
                               # routine same-day fake claim (which is meant
                               # to represent a "simultaneous morning
                               # announcement" the public can't put a
                               # reliable order on), a claim made THIS late
                               # is unambiguously, observably later than the
                               # real Medium's/Seer's mandatory Day-1 reveal
                               # (which always happens first, before
                               # nominations even exist) -- so it's a
                               # structurally certain fake claim, not just a
                               # suspicious one, for EITHER role now that
                               # both are forced to reveal Day 1 (see
                               # seer_day_co). See village_ai's
                               # _is_impossible_late_forced_claim, which
                               # this list feeds into.
vote_history = None  # {day: {voter_idx: target_idx}} -- EVERY living player's
                      # individual vote each day (see run_vote). Nobody's forced
                      # into a single collective decision any more; the person
                      # actually executed is whoever got the most votes.
missed_wolf_vote_count = None  # {player_idx: count} -- persistent across the
                                # whole game. Incremented once per player, per
                                # day, whenever a TRUSTWORTHY Medium report later
                                # reveals that day's executed player really was a
                                # wolf and this player voted for someone else
                                # instead. Only ever suggestive evidence (see
                                # village_ai's suspicious_non_vote_count feature
                                # docstring for why this is learned, not a hard
                                # rule), so it's exposed as a plain count for the
                                # village's execution/vote policy to weigh on its
                                # own, not used to hard-confirm anything here.
seer_investigations = None  # PRIVATE list of {day, target, is_wolf} -- the real
                             # Seer's own night-investigation results, in order,
                             # known only to the Seer until they reveal (see
                             # seer_report). Never exposed via build_public_state.
                             # Normally day >= 1 (a real night's investigation),
                             # but ALWAYS starts with exactly one day=0 entry --
                             # see _assign_seer_random_white -- a real-world
                             # Werewolf variant rule ("ランダム白") where the
                             # game itself, not the Seer's own action, hands them
                             # one confirmed-non-wolf player for free the moment
                             # roles are dealt, before Day 1 even begins. day=0
                             # sorts before any real investigation, so
                             # seer_report's catch-up always reports it first.
seer_reports = None  # PUBLIC list of {claimant, day, player, is_wolf} -- daily
                      # investigation results from anyone claiming Sy, real OR
                      # fake, disambiguated only by "claimant" the same way
                      # nominations are (see medium's "nominations" dict) -- no
                      # trust gating happens here, that's left entirely to
                      # village_ai's learned suspicion features (see
                      # is_accused_by_sole_seer / is_accused_by_confirmed_real_
                      # seer -- both fully learned since Step 2, no hard rule).
self_declared_non_village = None  # PUBLIC list of player indices who have
                      # openly claimed a non-village identity ("狼CO" or
                      # "狂人CO") -- derived from endgame_claims (below)
                      # whenever the claim isn't "村人CO". Populated by any
                      # of the Madman's genuine endgame reveal
                      # (madman_endgame_reveal), a genuine Villager's decoy
                      # bluff (villager_endgame_bluff), or a wolf's own
                      # endgame declare (wolf_endgame_declare) -- all three
                      # write into this SAME list so none of them are
                      # distinguishable from the outside just by appearing
                      # in it (see config.py's comment on "Kg", Steps 4-5).
endgame_claims = None  # PUBLIC dict {player_idx: "village"/"wolf"/"madman"} --
                      # the specific identity claimed by each of the three
                      # endgame-declaration functions below, richer than the
                      # plain self_declared_non_village list above. Lets
                      # village_ai apply the exact same sole/contested-claim
                      # deduction a Ps/Hn/Sy claim already gets: at most ONE
                      # player can genuinely be the Madman (config.ROLES),
                      # so 2+ simultaneous "claim_madman" entries prove at
                      # least one of them is lying, the same way a contested
                      # Sy claim does -- see village_ai's is_sole_kg_claimant/
                      # is_contested_kg_claimant.
nominator_inconsistency = None  # {nominator_idx: times they voted differently from their OWN nomination target}
not_voting_own_black = None  # {seer_claimant_idx: days they did not vote any of their own black targets}
voted_for_protected_count = None  # {player_idx: count} -- persistent across the
                                   # whole game. Incremented once per player, per
                                   # day, whenever that player's vote targeted
                                   # someone village_ai.apply_protection_floor
                                   # would have excluded (a sole/uncontested
                                   # claimant, an already-confirmed villager, or
                                   # the confirmed real Hunter). A genuine
                                   # village-aligned voter has no reason to do
                                   # this -- their OWN execution policy already
                                   # excludes such candidates from its pool --
                                   # so this is the mirror image of
                                   # defied_sole_nomination_count: that one
                                   # catches failing to comply with a
                                   # trustworthy call, this one catches actively
                                   # voting to kill someone that call (or the
                                   # protection floor generally) already
                                   # vouches for. Wolves have no equivalent
                                   # restraint in wolf_ai.vote_policy, so this
                                   # is exactly the kind of vote a wolf
                                   # occasionally casts that a real villager
                                   # essentially never would.
off_line_voters = None  # {player_idx: count} -- days this player voted outside the roll candidates (see village_ai.off_line_vote_count)
defied_nomination_count = None  # {player_idx: count} -- persistent across the
                                 # whole game. Incremented once per player, per
                                 # day, whenever an (AT THAT MOMENT) uncontested
                                 # sole Medium claimant has a standing nomination
                                 # target who is still alive, and this player
                                 # voted for someone else that day instead. Unlike
                                 # missed_wolf_vote_count, this needs no later
                                 # confirmation to compute -- a sole Medium's
                                 # nomination is trustworthy public information
                                 # the moment it's made, matching the real-world
                                 # convention that ignoring one without visible
                                 # reason invites suspicion ("制裁吊り").


def reset_game():
    """
    Reset all per-game mutable state.

    Must be called before each run_game() so that repeated calls
    (e.g. thousands of self-play games) don't leak state from the
    previous game.
    """
    global board, co_history, history, guard_target, last_guard_saved
    global last_execution_target, medium_reports, guard_ever_succeeded
    global guard_success_count, guard_success_targets, claim_day, attack_victims, last_guard_choice, last_night_victim, _report_snapshot
    global guard_instructions, confirmed_fake_hunters, confirmed_real_hunter
    global fake_declarations, nominations, emergency_reveals, emergency_wolf_claims, fake_medium_reports, medium_claims
    global execution_history, vote_history, missed_wolf_vote_count, defied_nomination_count
    global voted_for_protected_count, seer_investigations, seer_reports, off_line_voters
    global nominator_inconsistency, not_voting_own_black
    global self_declared_non_village, endgame_claims
    global _pending_roll, _label_history, _day1_co_display_buffer, _day1_flushed
    global ROLES
    # 2026-10-03 "役欠け" (10a狩): this game's actual role list is a
    # fresh draw from config.ROLE_POOL_10 (one of the 8 non-wolf slots
    # dropped at random -- half the time that's a redundant extra
    # Villager and today's familiar 9-role composition comes out
    # unchanged; the other half, Kg/Sy/Ps/Hn is genuinely absent this
    # game). See config.draw_roles()'s docstring. len(ROLES) stays 9
    # either way, so nothing sized off it needs to change.
    ROLES = config.draw_roles()
    board = ["Gr"] * len(ROLES)
    co_history = [None] * len(ROLES)
    claim_day = [None] * len(ROLES)
    history = []
    guard_target = None
    last_guard_choice = None
    last_guard_saved = False
    guard_ever_succeeded = False
    guard_success_count = 0
    guard_success_targets = []
    attack_victims = []
    last_night_victim = None
    _report_snapshot = None
    _missed_counted_days.clear()
    last_execution_target = None
    execution_history = []
    _pending_roll = None
    _label_history = []
    _day1_co_display_buffer = []
    _day1_flushed = False
    medium_reports = []
    fake_medium_reports = []
    medium_claims = []
    guard_instructions = {}
    confirmed_fake_hunters = []
    confirmed_real_hunter = None
    fake_declarations = {}
    nominations = {}
    emergency_reveals = []
    emergency_wolf_claims = []
    vote_history = {}
    missed_wolf_vote_count = {}
    defied_nomination_count = {}
    off_line_voters = {}
    voted_for_protected_count = {}
    nominator_inconsistency = {}
    not_voting_own_black = {}
    seer_investigations = []
    seer_reports = []
    self_declared_non_village = []
    endgame_claims = {}
    _assign_seer_random_white()


def _assign_seer_random_white():
    """
    Real-world Werewolf variant rule ("ランダム白"): the moment roles are
    dealt (before Day 1 even starts), the GAME ITSELF -- not the Seer's
    own night action -- hands the Seer one confirmed-non-wolf player,
    chosen uniformly at random from everyone else. This costs no night
    action and is separate from seer_night_investigate's own proactive
    ability; it's recorded as a day=0 entry in seer_investigations so
    it sorts first whenever the Seer eventually reveals and catches up
    on their results (see seer_report).

    Called at the end of reset_game(), so it always runs exactly once
    per game, before anyone has acted.
    """
    seer_index = next((i for i, role in enumerate(ROLES) if role == "Sy"), None)
    if seer_index is None:
        return

    # Every player is still alive at this point (board was just reset)
    # -- no need to filter by board state, only by role and self.
    pool = [i for i in range(len(ROLES)) if i != seer_index and ROLES[i] != "Wo"]
    if not pool:
        return

    target = random.choice(pool)
    seer_investigations.append({"day": 0, "target": target, "is_wolf": False})


reset_game()


# ============================================================
# Board Display
# ============================================================

def show_board():
    alive = [PLAYERS[i] for i in range(len(PLAYERS)) if board[i] != "--"]
    dead = [PLAYERS[i] for i in range(len(PLAYERS)) if board[i] == "--"]
    print(f"生存({len(alive)}): {' '.join(alive)}" + (f" / 死亡: {' '.join(dead)}" if dead else ""))


# 2026-10-03 user request: the 配役 line used to list players in raw
# seat order (P1..P9), which was the SAME as role order back when
# ROLES never changed -- but now that config.draw_roles() reshuffles
# the role list every game (the "役欠け" feature), seat order and role
# order have nothing to do with each other, and scanning a shuffled
# line to notice "wait, there's no Sy this game" is slow. Fixing the
# DISPLAY order to always be 狼→狂→占→霊→狩→村, independent of seating,
# makes a missing role jump out immediately as a gap in this sequence.
# Players sharing a role still sort by their own number (P1 before P9).
_ROLE_DISPLAY_ORDER = ["Wo", "Kg", "Sy", "Ps", "Hn", "Vi"]


def show_roles():
    """Reference table of true roles, shown once at game start, grouped
    by role in the fixed order 狼→狂→占→霊→狩→村 (see _ROLE_DISPLAY_ORDER)
    rather than by seat/player index, so a role that's absent this game
    (config.draw_roles()'s "役欠け") is immediately visible as a gap
    instead of being hidden in a shuffled player order."""
    order_index = {role: i for i, role in enumerate(_ROLE_DISPLAY_ORDER)}
    indices = sorted(
        range(len(PLAYERS)),
        key=lambda i: (order_index.get(ROLES[i], len(_ROLE_DISPLAY_ORDER)), i),
    )
    print("配役: " + "  ".join(f"{PLAYERS[i]}({ROLE_JP[ROLES[i]]})" for i in indices))


# ============================================================
# Medium's passive report (day AFTER an execution)
# ============================================================

def _note_medium_claim(claimant, exec_day, player, claimed_is_wolf):
    """
    Record one spoken Medium verdict in the PUBLIC, symmetric claims
    list (same shape for a real Medium and a fake one -- nothing about
    which list it lands in says who is genuine). If this claimant is
    the ONLY Medium claimant so far and the verdict is "wolf", anyone
    who voted for someone else that day gets a missed_wolf_vote_count
    (the same human "you didn't vote the wolf" suspicion the trusted
    path used to give only for a genuine Medium).
    """
    medium_claims.append({
        "claimant": claimant, "day": exec_day,
        "player": player, "is_wolf": claimed_is_wolf,
    })
    ps_claimants = [i for i, co in enumerate(co_history) if co == "Ps"]
    if claimed_is_wolf and len(ps_claimants) == 1 and ps_claimants[0] == claimant:
        _count_missed_wolf_votes(exec_day, player)



def _count_missed_wolf_votes(exec_day, player):
    """Each execution day adds to missed_wolf_vote_count at most ONCE
    (2026-10-06: the sole-claim path and the later trusted backfill
    could both count the same day)."""
    if exec_day in _missed_counted_days:
        return
    _missed_counted_days.add(exec_day)
    for voter, target in vote_history.get(exec_day, {}).items():
        if target != player:
            missed_wolf_vote_count[voter] = missed_wolf_vote_count.get(voter, 0) + 1


def medium_report(day, verbose=True):
    """
    If the Medium is alive and has revealed, they automatically
    learn (and announce) whether YESTERDAY's executed player was
    a Werewolf. This is the Medium's actual ability -- previously
    the role only had the CO decision and nothing to show for it.

    The verdict is ALWAYS announced out loud (see the verbose print
    below, unconditional on trust) -- exactly what a real Medium
    player would do every single day, contested or not. Whether it
    gets RECORDED into medium_reports (and used by other deduction)
    is a separate question, gated on "trustworthy" below, and once
    that gate opens, every previously-unrecorded day gets backfilled
    too -- see the comments further down for why that backfill is
    necessary and how it stays leak-free.
    """

    global confirmed_real_hunter, missed_wolf_vote_count

    if last_execution_target is None:
        return

    medium_index = next(
        (i for i, role in enumerate(ROLES) if role == "Ps"), None
    )

    if medium_index is None:
        return

    if board[medium_index] == "--" or co_history[medium_index] != "Ps":
        return  # dead, or hasn't revealed -- no public report

    # CRITICAL: the report we're about to announce is only
    # trustworthy PUBLIC information if nobody could mistake it for
    # someone else's. When 2+ players have claimed Medium, an
    # outside observer (and this AI) has no legitimate way to know
    # WHICH claimant's verdict is the real one -- the guard-
    # instruction trick lets Hunter claims get disambiguated via an
    # observable in-game signal (whose assigned target the wolves
    # actually killed), but Medium has no such mechanism of its own.
    # Simulating "the true medium secretly knows this" and then
    # broadcasting it as settled fact regardless of contest state
    # would hand the AI free, unearned ground truth -- exactly the
    # kind of oracle leak we removed for Hunter via num_hunter_claimants
    # gating.
    #
    # BUT the contest doesn't have to stay unresolved forever, and it
    # doesn't have to be resolved by THIS ability either. Once some
    # OTHER already-legitimate public deduction (the Hunter's guard-
    # verification "confirmed white" list, the wolf-counting pigeonhole
    # deduction, etc. -- exactly what build_public_state's
    # confirmed_villagers already collects) has cleared all but one of
    # today's Day-1 Ps claimants of being the wolf, the ambiguity is
    # gone: whichever single claimant is still unresolved is the only
    # one who could possibly be lying, so there's no oracle leak in
    # finally trusting the (always-computed-truthfully) report from
    # this point on -- it's the same kind of "public knowledge someone
    # else already earned" that guard-instruction verification itself
    # relies on. This is what makes a direct contradiction like "the
    # newly-cleared claimant says 白, the still-unresolved rival says
    # 黒 on the same person" actually catchable by
    # village_ai._is_confirmed_contradicting_medium, instead of that
    # feature staying permanently unreachable for as long as any
    # contest is technically still open.
    #
    # A Ps claim made on any day OTHER than Day 1 is a structural
    # impossibility for the real Medium (medium_day_co forces them to
    # reveal Day 1, unconditionally) -- so it's already PROVEN fake on
    # its own, with total certainty, no comparison needed. A claim in
    # that category must not be allowed to "contest" a genuine Day-1
    # claimant and thereby poison trust in their reports -- that's
    # exactly backwards, and it's what let a wolf's late, obviously-
    # fake Ps claim silently strip a confirmed-real Day-1 Medium's
    # trustworthy reports (and eventually their execution protection
    # in village_ai) for the rest of the game, even after that late
    # claimant was caught and executed. co_history never clears a
    # dead player's claim, so this exclusion must hold regardless of
    # whether that claimant is still alive.
    day1_medium_claimants = [
        i for i, co in enumerate(co_history)
        if co == "Ps" and claim_day[i] == 1
    ]
    confirmed_villagers_now = build_public_state(day)["confirmed_villagers"]
    unresolved_medium_claimants = [
        i for i in day1_medium_claimants if i not in confirmed_villagers_now
    ]
    # 2026-10-05 information-asymmetry fix: the TRUE verdict is only
    # recorded as settled fact once the Medium is IDENTITY-proven real
    # by public logic (confirmed_real_claimants). The old gate -- "at
    # most one unresolved Day-1 claimant" -- also fired for a SOLE
    # claimant, which meant a real lone Medium's reports were recorded
    # as truth while a fake lone Medium's were not: the village could
    # tell them apart, which no human can.
    trustworthy = medium_index in build_public_state(day)["confirmed_real_claimants"]

    is_wolf = ROLES[last_execution_target] == "Wo"
    _note_medium_claim(medium_index, day - 1, last_execution_target, is_wolf)

    if verbose:
        verdict = "黒" if is_wolf else "白"
        print(f"【霊能結果】{PLAYERS[medium_index]}: {PLAYERS[last_execution_target]}"
              f"は{verdict}でした")

    if not trustworthy:
        return

    # BACKFILL, not just today: while the Day-1 Medium contest was
    # still unresolved (a wolf or the Madman also claimed "Ps" on Day
    # 1, and nothing had cleared either one yet), this function had
    # NOTHING legitimate to record for any of those earlier days --
    # recording the real Medium's report as settled fact while the
    # contest was still technically open would have been exactly the
    # oracle leak the comment above warns about. But once the contest
    # genuinely resolves (some OTHER already-legitimate public
    # deduction clears all but one claimant), every one of the real
    # Medium's PAST daily verdicts becomes exactly as legitimate as
    # today's -- and, crucially, none of that history is lost: each
    # past day's verdict is recomputable with total precision from
    # execution_history (who was actually executed that day) plus
    # ROLES (who actually is/isn't a wolf), so no separate hidden
    # "private investigation log" needs to have been kept for the
    # Medium the way seer_investigations already is for the Seer.
    #
    # Without this backfill, a Madman contesting "Ps" on Day 1 could
    # permanently erase the real Medium's information for the ENTIRE
    # game just by making the contest never resolve, even while the
    # real Medium stayed alive and kept correctly reporting out loud
    # every single day (see the verbose print above, which -- exactly
    # like a real Medium player's own claim -- is never itself gated
    # on trustworthy) -- the information was always publicly SPOKEN,
    # it just wasn't being RECORDED anywhere a feature could use it.
    # That silently starved village_ai._is_confirmed_contradicting_
    # medium of the one thing it needs to catch a fake Medium's
    # contradicting lies, for as long as the contest stayed open --
    # which, since Medium (unlike Hunter) has no verification
    # mechanism of its own, could easily mean the whole game.
    # NOTE ON DAY NUMBERING: entries here are tagged with the
    # EXECUTION's own day (matching execution_history's indexing),
    # not the day_phase call day that happened to report it -- e.g.
    # today's report (about YESTERDAY's execution) is tagged
    # `day - 1`, not `day`. Nothing currently reads medium_reports'
    # "day" field for anything (see this file's global declaration
    # comment), but keeping ONE consistent convention here matters
    # for already_reported_days' dedup below: mixing "the day this was
    # CALLED" and "the day this was EXECUTED" numbering in the same
    # set would make a day's entry fail to dedup against itself.
    today_exec_day = day - 1
    already_reported_days = {r["day"] for r in medium_reports}
    for d in range(1, today_exec_day):
        if d in already_reported_days:
            continue
        target_d = execution_history[d - 1]
        is_wolf_d = ROLES[target_d] == "Wo"
        medium_reports.append({
            "claimant": medium_index,
            "day": d,
            "player": target_d,
            "is_wolf": is_wolf_d,
        })
        if verbose:
            verdict_d = "黒" if is_wolf_d else "白"
            print(f"   → (遡って確定) {d}日目の霊能結果: "
                  f"{PLAYERS[medium_index]}: {PLAYERS[target_d]}は{verdict_d}でした")
        if is_wolf_d and len([i for i, co in enumerate(co_history) if co == "Ps"]) > 1:
            _count_missed_wolf_votes(d, target_d)

    medium_reports.append({
        "claimant": medium_index,
        "day": today_exec_day,
        "player": last_execution_target,
        "is_wolf": is_wolf,
    })

    # Voting-based suspicion tracking: we now KNOW, with the same
    # certainty the rest of this function already relies on, that
    # yesterday's execution really did remove a wolf. So anyone
    # who voted for someone else that day had exactly the same
    # public information available and still didn't catch it --
    # worth remembering as one data point (not proof of anything
    # -- a genuine villager can easily have suspected someone else
    # for good reason that day too). See village_ai's
    # suspicious_non_vote_count feature for how this gets used.
    if is_wolf and len([i for i, co in enumerate(co_history) if co == "Ps"]) > 1:
        _count_missed_wolf_votes(today_exec_day, last_execution_target)

    # REMOVED: this used to try to deduce the status of OTHER
    # claimants of the same role from today's verdict, on the
    # assumption that "only wolves ever fake a claim in this game."
    # That assumption stopped being true back in Step 3 (the Madman
    # gained the ability to fake-claim a role too), and this block
    # was never updated to account for it -- it's the exact same
    # class of bug already found and fixed once this session in
    # _wolf_counting_deduction, just in a different spot:
    #
    #   - executed claimant was a wolf -> the old code concluded the
    #     sole remaining OTHER claimant must be "confirmed real by
    #     elimination." But that remaining claimant could just as
    #     easily be the Madman's fake claim -- knowing the executed
    #     claimant was specifically a WOLF (rather than "not the
    #     genuine holder" in general) says nothing about who else is
    #     lying.
    #   - executed claimant was NOT a wolf -> the old code concluded
    #     they must have been the genuine holder, so every OTHER
    #     claimant is "definitely fake." But "not a wolf" does not
    #     mean "genuine" -- the executed claimant could itself have
    #     been the Madman's fake claim, in which case the real
    #     holder is still out there among the "other claimants."
    #
    # Both directions could mislabel a Madman as confirmed_real_hunter
    # (permanent execution protection in village_ai, PLUS laundering
    # that Madman's claimed guard targets straight into
    # confirmed_villagers as ground truth) or mislabel the genuine
    # Hunter as confirmed_fake_hunters (feeding confirmed_dead_wolves,
    # per estimate_wolf_alive_distribution's own docstring caveat).
    #
    # Per the same "Step 2" precedent the rest of this codebase
    # already follows (hard certainties that turned out to be
    # Madman-unsafe get replaced with learned probabilistic signals,
    # not patched with more special cases), this deduction is simply
    # removed rather than fixed in place. The genuinely safe part of
    # what it was trying to capture -- "this executed claimant is
    # now confirmed non-wolf" / "...confirmed wolf" -- is already
    # real information and already flows into confirmed_villagers /
    # medium_reports above without needing this extra step. What's
    # left uncertain (whether the OTHER same-role claimants are real
    # or fake) is exactly what village_ai's generic, learned
    # Hn_contest_wolf_prob / is_claimant features are for -- they
    # degrade gracefully to "still uncertain" instead of asserting a
    # false certainty. The one REMAINING legitimate way to actually
    # prove a Hunter claimant real/fake with total certainty is
    # resolve_guard_instructions() below, which relies on an
    # observed guard-instruction OUTCOME (whose assigned watch target
    # actually survived/died), not on an assumption about who is
    # capable of faking a claim.


_report_snapshot = None  # public state frozen BEFORE this morning's reports


def _begin_report_snapshot(day):
    """2026-10-06: every morning report (real or fake) is made
    SIMULTANEOUSLY. A fake claimant's decision must not see what another
    claimant said today (that would be a post-hoc lie). The snapshot is
    the public state before ANY of today's reports."""
    global _report_snapshot
    _report_snapshot = {"day": day, "state": build_public_state(day)}


def _end_report_snapshot():
    global _report_snapshot
    _report_snapshot = None


def _report_state(day):
    """Public state for a fake report decision: the pre-report snapshot,
    with only the (sequentially public) board / CO list refreshed."""
    if _report_snapshot is not None and _report_snapshot["day"] == day:
        st = dict(_report_snapshot["state"])
        fresh = build_public_state(day)
        for k in ("co_history", "board", "claim_day", "day1_config", "alive_count"):
            if k in fresh:
                st[k] = fresh[k]
        return st
    return build_public_state(day)


def _fake_medium_ctx(i, role, day):
    alive = sum(1 for b in board if b != "--")
    others = [j for j, co in enumerate(co_history)
              if co == "Ps" and j != i and board[j] != "--"]
    prior = [r for r in medium_claims if r["day"] < day - 1]
    exec_co = co_history[last_execution_target]
    teammate_ps = 0.0
    if role == "Wo":
        teammate_ps = 1.0 if any(
            j != i and ROLES[j] == "Wo" and co_history[j] == "Ps" and board[j] != "--"
            for j in range(len(ROLES))) else 0.0
    return [
        1.0 if (role == "Wo" and ROLES[last_execution_target] == "Wo") else 0.0,
        1.0 if others else 0.0,
        1.0 if any(r["claimant"] != i and r["is_wolf"] for r in prior) else 0.0,
        1.0 if any(r["claimant"] == i and r["is_wolf"] for r in prior) else 0.0,
        1.0 if exec_co in ("Sy", "Hn", "Ps") else 0.0,
        teammate_ps,
        1.0 if alive <= 5 else 0.0,
        day / 7.0,
    ]


def fake_medium_report(day, verbose=True):
    """
    Any FAKE Medium claimant -- a wolf OR the Madman (Step 3: the
    Madman can now bluff Ps/Hn/Sy exactly like a wolf can) -- who is
    alive and revealed ALSO gives a daily verdict on yesterday's
    execution -- a lie, since neither has any real ability. They
    choose whether to agree with the real Medium's verdict
    (consistent-looking, but gives the village nothing new to work
    with) or contradict it (creates a "split" -- risky, since it can
    get them caught, but muddies the water if the village can't
    otherwise tell who's genuine). Both sides share the exact same
    option set and feature shape (see madman_ai.choose_fake_report_
    strategy's docstring for why this one choice never needed any
    wolf-private teammate information in the first place) but score
    it with their OWN separate weights, so wolves and the Madman can
    learn different agree/contradict tendencies.
    """

    if last_execution_target is None:
        return

    true_is_wolf = ROLES[last_execution_target] == "Wo"

    for i, role in enumerate(ROLES):

        if role not in ("Wo", "Kg") or board[i] == "--" or co_history[i] != "Ps":
            continue

        side_name = "wolf" if role == "Wo" else "madman"
        module = wolf_ai if role == "Wo" else madman_ai

        strategy_state = _report_state(day)
        strategy_state["fake_ctx"] = _fake_medium_ctx(i, role, day)
        strategy = module.choose_fake_report_strategy(strategy_state)
        log_decision(side_name, "fake_report_strategy", strategy_state, strategy, day=day)

        if role == "Wo":
            # A wolf legitimately knows the truth about its teammate.
            reference = true_is_wolf
        else:
            # 2026-10-06 (M5, revised after re-audit): the Madman knows
            # nothing private, and a same-day claim list always has the
            # real Medium first -- so using "what another claimant just
            # said" would hand it the true verdict. Reports are treated
            # as simultaneous: its only reference is the default white.
            reference = False
        claimed_is_wolf = reference if strategy == "agree" else not reference

        fake_medium_reports.append({
            "claimant": i,
            "day": day,
            "player": last_execution_target,
            "is_wolf": claimed_is_wolf,
        })
        _note_medium_claim(i, day - 1, last_execution_target, claimed_is_wolf)

        if verbose:
            verdict = "黒" if claimed_is_wolf else "白"
            print(f"【霊能結果】{PLAYERS[i]}: {PLAYERS[last_execution_target]}"
                  f"は{verdict}でした")


def seer_report(day, verbose=True):
    """
    If the real Seer is alive and has revealed, publish every one of
    their private night-investigation results that hasn't been made
    public yet -- catching up in one go on anything investigated
    BEFORE they revealed (a real Seer who stays silent for a few days
    still investigates every night; those early results were simply
    never public until now), then continuing one per night from here.

    Unlike medium_report, there's no "trustworthy" gate here: each
    entry is tagged with its own claimant (see seer_reports'
    declaration), so a contested claim doesn't need to be resolved
    before this can be recorded -- it's exactly the same "always
    record, let the trust live in the features" approach main.py
    already uses for Medium nominations (nominations is never gated
    on contest state either). What DOES need the claimant's identity
    to be settled is treating a report as GROUND TRUTH (see
    build_public_state's confirmed_real_seers cascade) -- this
    function only ever appends the real Seer's own genuinely true
    results, it just doesn't presume anyone else can already tell
    that from the outside.
    """

    seer_index = next((i for i, role in enumerate(ROLES) if role == "Sy"), None)
    if seer_index is None:
        return

    if board[seer_index] == "--" or co_history[seer_index] != "Sy":
        return  # dead, or hasn't revealed -- nothing public to report yet

    # NOTE: day_phase calls this function TWICE on the same day -- once
    # before today's CO decisions (catching up anyone who already
    # revealed on an earlier day) and once again right after
    # seer_day_co. That second call matters: without it, a Seer who
    # decides to reveal TODAY wouldn't have anything to report until
    # TOMORROW's day_phase, since this function only reports for
    # claimants who "already have revealed" as of the moment it runs
    # -- and if that Seer is killed tonight, tomorrow's day_phase
    # never happens for them, so even their day-0 free "random white"
    # result (known since before the game started) would be silently
    # lost forever, both from the printed log AND from seer_reports
    # (i.e. it would never even become usable ground truth for
    # build_public_state's confirmed_real_seers cascade). The
    # already_reported_days dedup below is what makes calling this
    # function twice in the same day_phase safe.
    already_reported_days = {
        r["day"] for r in seer_reports if r["claimant"] == seer_index
    }
    pending = [inv for inv in seer_investigations if inv["day"] not in already_reported_days]

    for inv in sorted(pending, key=lambda x: x["day"]):
        seer_reports.append({
            "claimant": seer_index,
            "day": inv["day"],
            "player": inv["target"],
            "is_wolf": inv["is_wolf"],
        })
        if verbose:
            verdict = "黒" if inv["is_wolf"] else "白"
            print(f"【占い結果】{PLAYERS[seer_index]}: {PLAYERS[inv['target']]}"
                  f"は{verdict}でした")


def fake_seer_report(day, verbose=True):
    """
    Any FAKE Seer claimant -- a wolf OR the Madman (Step 3) -- who is
    alive and revealed also produces a daily "investigation result"
    -- entirely fabricated, since neither has any real ability.
    Unlike a fake Medium report (which only needs agree/contradict
    against one already-known verdict), this has to invent BOTH a
    target and a verdict every day. A wolf's option set includes
    vouching for a known TEAMMATE as clean (wolf_ai.
    choose_fake_seer_report_strategy); the Madman has no teammate to
    vouch for, so it gets its own, shorter option list instead (see
    madman_ai.choose_fake_seer_report_strategy / MADMAN_FAKE_SEER_
    REPORT_OPTIONS).

    NOTE: day_phase calls this function TWICE on the same day -- once
    before today's CO decisions (for anyone who revealed on an
    earlier day) and once again right after wolf_day_co/madman_day_co
    (so a claimant who reveals Sy TODAY doesn't have to wait until
    tomorrow's day_phase for their first fabricated report -- see
    seer_report's matching NOTE for why that lag mattered enough to
    fix). The per-day dedup check right below is what makes calling
    this twice in one day safe -- without it, an already-revealed
    claimant would get two fabricated reports on every day after
    their first.
    """

    for i, role in enumerate(ROLES):

        if role not in ("Wo", "Kg") or board[i] == "--" or co_history[i] != "Sy":
            continue

        if any(r["claimant"] == i and r["day"] == day - 1 for r in seer_reports):
            continue  # already produced today's report on the earlier call

        strategy_state = _report_state(day)

        rivals = [
            j for j, co in enumerate(co_history)
            if co == "Sy" and j != i and board[j] != "--"
        ]
        others = [j for j in range(len(ROLES)) if board[j] != "--" and j != i]
        # 2026-10-06 (round 4): a REAL Seer investigates before the night
        # attack, so last night's victim is a possible report target. A
        # fake must be able to name them too, or "reported someone who
        # died overnight" would identify the real Seer.
        if last_night_victim is not None and last_night_victim != i and last_night_victim not in others:
            others.append(last_night_victim)

        # On Day 1, choose_fake_seer_report_strategy (wolf_ai's and
        # madman_ai's) already narrows its own option list to white-
        # only outcomes -- a claimant's first-ever public report always
        # stands in for the mandatory pre-game "random white"
        # investigation (_assign_seer_random_white), which the real
        # Seer can never report as black. So `strategy` below is
        # guaranteed to resolve through a `False` (white) entry in
        # pool_map whenever `day == 1`; nothing further needs checking
        # here.
        if role == "Wo":
            side_name = "wolf"
            strategy = wolf_ai.choose_fake_seer_report_strategy(strategy_state)
            teammates = [
                j for j, r in enumerate(ROLES)
                if r == "Wo" and board[j] != "--" and j != i
            ]
            # Teammates are excluded from the accuse-rival pool up
            # front: a wolf slandering its OWN side is a strict-
            # dominance loss with no scenario where it helps, so it's
            # filtered out structurally rather than left for training
            # to (slowly, noisily) discover on its own -- see
            # wolf_ai.choose_accuse_target's docstring. What's left in
            # accuse_rivals is only genuine candidates: the real Seer,
            # or a Madman who also happened to claim Sy.
            accuse_rivals = [j for j in rivals if j not in teammates]
            pool_map = {
                "accuse_rival_black": (accuse_rivals, True),
                "vouch_teammate_white": (teammates, False),
                "random_black": (others, True),
                "random_white": (others, False),
            }
        else:
            side_name = "madman"
            strategy = madman_ai.choose_fake_seer_report_strategy(strategy_state)
            accuse_rivals = rivals  # no teammates for a Madman to exclude
            pool_map = {
                "accuse_rival_black": (accuse_rivals, True),
                "random_black": (others, True),
                "random_white": (others, False),
            }

        log_decision(side_name, "fake_seer_report_strategy", strategy_state, strategy, day=day)

        pool, is_wolf = pool_map[strategy]

        # 2026-10-06 (A4 / H4): the same structural limits the REAL Seer
        # works under, applied to a fake one -- never name someone it
        # already reported on, never black-accuse a publicly proven
        # non-wolf, never white a publicly proven wolf, and (wolf) never
        # slander its own teammate. Keeps self-inflicted contradictions
        # from being a real-vs-fake tell and from being a free "lie
        # detector" signal. Falls back to the unfiltered pool if empty.
        _already = {r["player"] for r in seer_reports if r["claimant"] == i}
        _proven_white = set(strategy_state.get("confirmed_villagers", []))
        _proven_black = set(strategy_state.get("confirmed_wolves", [])) | {
            r["player"] for r in strategy_state.get("medium_reports", []) if r["is_wolf"]}

        def _legal(c):
            if c in _already:
                return False
            if is_wolf and (c in _proven_white or (role == "Wo" and ROLES[c] == "Wo")):
                return False
            if (not is_wolf) and c in _proven_black:
                return False
            return True
        _filtered = [c for c in pool if _legal(c)]
        if _filtered:
            pool = _filtered
        # WHICH accuse_rivals candidate to name is itself a learned
        # decision (see wolf_ai.choose_accuse_target) -- but only when
        # there's a genuine choice to make: the preferred pool must be
        # non-empty (checked here, before the generic empty-pool
        # fallback below can silently swap it out for `others`) and
        # have more than one candidate (a single candidate needs no
        # decision, and LinearPolicy.update's gradient would be zero
        # anyway -- see MADMAN_FAKE_SEER_REPORT_DAY1_OPTIONS's comment
        # on the same non-choice case).
        if not pool:
            # Preferred pool (a rival / a teammate) doesn't exist right
            # now -- fall back to a random target with the SAME verdict
            # polarity the chosen strategy implied.
            pool = others
            is_wolf = strategy in ("accuse_rival_black", "random_black")
        if not pool:
            continue

        if len(pool) > 1:
            target_state = dict(strategy_state)
            target_state["candidates"] = pool
            target_state["fake_polarity"] = bool(is_wolf)
            target_state["self_index"] = i
            if role == "Wo":
                target_state["wolf_indices"] = [
                    j for j, r in enumerate(ROLES) if r == "Wo" and board[j] != "--"]
            _vs = _report_state(day)
            _vs["candidates"] = [j for j in range(len(ROLES)) if board[j] != "--"]
            target_state["village_vote_probs"] = village_ai.execution_distribution(_vs)
            chooser = (
                wolf_ai.choose_accuse_target if side_name == "wolf"
                else madman_ai.choose_accuse_target
            )
            target = chooser(target_state)
            log_decision(side_name, "fake_seer_accuse_target", target_state, target, day=day)
        else:
            target = random.choice(pool)

        # L1 (2026-10-06): same day-tag convention as the real Seer's
        # reports (investigation night = day - 1; Day 1 -> 0).
        seer_reports.append({
            "claimant": i, "day": day - 1, "player": target, "is_wolf": is_wolf,
        })

        if verbose:
            verdict = "黒" if is_wolf else "白"
            print(f"【占い結果】{PLAYERS[i]}: {PLAYERS[target]}は{verdict}でした")


# ============================================================
# Medium / Hunter CO (asked once per day, while still silent)
# ============================================================

def _make_nomination(claimant_idx, day, verbose=True):
    """
    Have a Medium claimant (real OR fake) publicly name one living
    player (other than themselves) as a suspect ("指定"). Shared by
    the real Medium's forced Day 1 reveal and a wolf's fake Medium
    claim, so both use the exact same nomination_strategy learned
    choice.

    The candidate pool used to be restricted to players with NO CO at
    all (co_history[j] is None) -- i.e. only "flat gray" suspects. But
    when TWO players both claim Medium, the single most natural target
    for either one to name is the OTHER Medium claimant: exactly one
    of them has to be lying, so pointing at the rival is at least as
    informative a nomination as pointing at an uninvolved silent
    player, and a real human Medium would absolutely consider it. The
    old restriction made that structurally impossible -- a contested
    co-claimant could never even be offered as a candidate, no matter
    how suspicious build_public_state's own features already judged
    them to be. Now every other living player is a legal candidate,
    and it's left to the learned nomination_target_policy (which sees
    the exact same suspicion features village_ai's execution vote
    does, including the contested-claim ones) to decide whether a
    rival claimant or a genuinely uninvolved player is the better call.
    """

    candidates = [
        j for j in range(len(ROLES))
        if board[j] != "--" and j != claimant_idx
    ]
    if not candidates:
        return

    nom_state = build_public_state(day)
    nom_state["voter"] = claimant_idx
    if ROLES[claimant_idx] == "Ps":
        nom_state["voter_private_truth"] = {
            t: (ROLES[t] == "Wo") for t in execution_history if t is not None}
    strategy = medium_ai.choose_nomination_strategy(nom_state)
    # 2026-10-06 (training-signal fix): a FAKE medium (wolf/madman)
    # samples from the same policy but is NOT logged -- it used to be
    # trained with the village's reward, which pushed the shared
    # medium policy in the wrong direction (opposite-sign samples).
    if ROLES[claimant_idx] == "Ps":
        log_decision("medium", "nomination_strategy", nom_state, strategy, day=day)

    if strategy == "nominate":
        nom_state["candidates"] = candidates
        target = medium_ai.choose_nomination_target(nom_state, candidates)
        if ROLES[claimant_idx] == "Ps":
            log_decision("medium", "nomination_target", nom_state, target, day=day)
        nominations[claimant_idx] = target
        if verbose:
            print(f"{PLAYERS[claimant_idx]}(霊CO)が{PLAYERS[target]}を指定")


def make_all_nominations(day, verbose=True):
    """
    After EVERYONE's CO decision for the day is in (real Medium,
    any fake Medium claims, Hunter claims, etc.), each Ps claimant
    who hasn't nominated yet gets to name a suspect. Doing this
    LAST -- rather than the instant each claimant reveals -- means
    a nomination can be informed by whether a competing claim
    showed up today too, matching how a real Medium player would
    actually wait to see the day's claims shake out before pointing
    at someone.
    """

    for i, co in enumerate(co_history):
        # 2026-10-06 (M6): also nominate AGAIN once the previous nominee is dead
        if co == "Ps" and board[i] != "--" and (
                i not in nominations or board[nominations[i]] == "--"):
            _make_nomination(i, day, verbose=verbose)


def medium_day_co(day, verbose=True):
    """
    The Medium is REQUIRED to RCO (reveal) first thing on Day 1 --
    matching standard human convention. This is no longer a learned
    choice; every game, the Medium reveals immediately if alive.
    (The old learned timing choice, medium_ai.choose_co_action, had
    zero call sites once this forced reveal landed and was removed
    from medium_ai.py in a later cleanup pass.)

    NOTE: the "灰指定" nomination itself is made LATER (see
    make_all_nominations, called after every side's CO decision for
    the day is in) -- not here. A real Medium player would wait to
    see whether anyone else claims the role too before deciding who
    to point at; nominating in the same breath as revealing, before
    knowing that, doesn't match how human play actually paces out.
    """

    if day != 1:
        return

    for i, role in enumerate(ROLES):

        if role == "Ps" and board[i] != "--" and co_history[i] is None:
            board[i] = "Ps"
            co_history[i] = "Ps"
            claim_day[i] = day
            announce_co(i, "Ps", verbose=verbose, day=day)


def hunter_day_co(day, verbose=True):

    for i, role in enumerate(ROLES):

        if role == "Hn" and board[i] != "--" and co_history[i] is None:

            state = build_public_state(day, blind=(day == 1))
            state["gray_count"] = sum(
                1 for j, s in enumerate(state["board"])
                if s != "--" and state["co_history"][j] is None
            )

            action = hunter_ai.choose_co_action(state)
            log_decision("hunter", "co_action", state, action, day=day)

            if action == "reveal":
                board[i] = "Hn"
                co_history[i] = "Hn"
                claim_day[i] = day

                announce_co(i, "Hn", verbose=verbose, day=day)
                if verbose and guard_success_targets:
                    saved = "、".join(PLAYERS[t] for t in guard_success_targets)
                    print(f"   → これまでの護衛先を公表: {saved} は確定で白!")


def seer_day_co(day, verbose=True):
    """
    The Seer is REQUIRED to RCO (reveal) first thing on Day 1 --
    matching standard human convention, same as the Medium
    (medium_day_co). This is no longer a learned choice; every game,
    the Seer reveals immediately if alive. (The old learned timing
    choice, seer_ai.choose_co_action, had zero call sites once this
    forced reveal landed and was removed from seer_ai.py in a later
    cleanup pass.)

    This used to be a genuinely open strategic tradeoff (see the
    now-outdated reasoning that lived in seer_ai's module docstring:
    a proactive, repeatable information source is a much higher-value
    kill target than a Medium's one-shot-per-execution report, so
    staying silent a while longer used to be a real option). Forcing
    Day-1 reveal for the Seer too, at the user's explicit request to
    match the standard house rule, has a second, structural payoff
    beyond just following convention: it makes a "Sy" claim on any
    day other than Day 1 JUST AS STRUCTURALLY IMPOSSIBLE as a late
    "Ps" claim already was -- see village_ai's
    _is_impossible_late_forced_claim, which now covers both roles
    identically instead of only the Medium.
    """

    if day != 1:
        return

    for i, role in enumerate(ROLES):

        if role == "Sy" and board[i] != "--" and co_history[i] is None:
            board[i] = "Sy"
            co_history[i] = "Sy"
            claim_day[i] = day
            announce_co(i, "Sy", verbose=verbose, day=day)


# ============================================================
# Game State (for AI modules)
# ============================================================
#
# AI modules (wolf_ai.py, village_ai.py, ...) should never read
# the global board/co_history/ROLES lists directly. Instead,
# main.py packages exactly what that side is allowed to know
# into a plain dict and passes it in. This keeps each AI module
# fully decoupled from main.py's internals.

def _wolf_counting_deduction(day, co_history_view=None, board_view=None):
    """
    A pigeonhole-counting argument: if k people claim the same
    role, at most 1 can be genuine, so AT LEAST (k-1) of them are
    FAKE. Summing this MINIMUM required-fake count across every
    claimed role gives a lower bound on how many fake-claim-capable
    players must be sitting among the claimants.

    If that lower bound EXACTLY equals the total number of players
    who could possibly BE one of these fake claimants, every one of
    them is now accounted for among the claimants -- which forces
    two further conclusions with total certainty, independent of
    guard-instruction verification or medium reports:

      - any role claimed by exactly 1 person (uncontested) is
        confirmed genuine, since 0 fakes are needed there and none
        are left over to hide there either.
      - every SILENT player (no claim at all) is confirmed a
        villager, since no fake-claim-capable player remains
        unaccounted for to be hiding there.

    BUG FIX: "the total number of players who could possibly BE one
    of these fake claimants" used to just be total_wolves, back when
    "nobody but a wolf has any reason to fake a claim" was still
    true. That stopped being true the moment the Madman (config.py's
    "Kg") was added -- it can fake-claim Ps/Hn/Sy exactly like a wolf
    can (see madman_ai.py), so it's a SECOND source of "excess"
    claimants this sum has to account for. Comparing the naive sum
    only against total_wolves means a game where a Madman's fake
    claim contributes to some role's excess count will have
    min_fakes_required PERMANENTLY OVERSHOOT total_wolves by exactly
    the number of Madman fake-claims made (since the Madman's excess
    contribution isn't a wolf) -- the sum can then never equal
    total_wolves again for the rest of the game, so this deduction
    silently stops firing AT ALL, forever, the instant that happens.
    That's not a rare edge case: it happens any time a Madman fake-
    claims a role that's ALSO claimed by anyone else (itself the
    common case for a role a real player has already revealed for) --
    e.g. 3 Ps claimants where 1 is a wolf, 1 is the real Medium, and
    1 is the Madman needs exactly 2 fakes, but only 1 of those 2 is a
    wolf, so total_wolves alone permanently undercounts by 1. The
    real-world symptom: EVERY silent player stays forever un-
    confirmable for the rest of that game, no matter how thoroughly
    the wolves and Madman get exposed elsewhere -- silently starving
    village_ai's execution policy of the "this player is provably
    clean, don't waste a vote here" signal it would otherwise have
    earned, right when a contested role-claim (with its far more
    informative, still-unresolved suspicion signal) should have been
    getting priority instead. Fixed by comparing against
    total_wolves + total_madmen -- the true maximum number of
    players who could ever contribute an "excess" claim -- which
    reduces to the exact original check whenever a game has no
    Madman (total_madmen == 0).

    Returns (extra_confirmed_villagers, extra_confirmed_real) as
    two lists of player indices. This is deliberately checked at
    the point of a HARD, 100%-certain deduction (a fact derivable
    from the game's own claim counts), not a learned heuristic --
    matching how is_impossible_late_medium_claim etc. are treated
    elsewhere. It still only ever proves "no fake remains OUTSIDE
    the claimed contests" -- it never says WHICH claimant within a
    still-contested role is the genuine one (a wolf's and a Madman's
    excess claims remain indistinguishable from each other here,
    which is fine: nothing above needs to tell them apart).
    """
    co_hist = co_history_view if co_history_view is not None else co_history
    brd = board_view if board_view is not None else board

    total_wolves = sum(1 for role in config.ROLE_POOL_10 if role == "Wo")
    # 2026-10-05 information-asymmetry fix: villagers do NOT know whether
    # this game's Madman exists (役欠け can drop it -- config.draw_roles),
    # so the deduction may only assume the POOL's maximum, never the
    # actual ROLES count. (Using the true count told the village whether
    # the Madman was absent.)
    total_madmen = sum(1 for role in config.ROLE_POOL_10 if role == "Kg")
    total_fake_capable = total_wolves + total_madmen

    # IMPORTANT: count every claim ever made, dead or alive. Who
    # claimed what is a historical fact that doesn't change when the
    # claimant is later executed or killed -- a dead Medium-claimant
    # who turned out to be a wolf is still one of the people who
    # claimed Medium. Filtering this down to only currently-alive
    # claimants was the original bug: once one of two co-claimants
    # died, the live count would drop to 1 (an "uncontested" claim,
    # needing 0 wolves), silently erasing a pigeonhole fact that was
    # already proven true and can never become false again.
    claim_counts = {}
    for co in co_hist:
        if co is not None:
            claim_counts[co] = claim_counts.get(co, 0) + 1

    min_fakes_required = sum(max(0, c - 1) for c in claim_counts.values())

    if min_fakes_required != total_fake_capable:
        return [], []

    extra_villagers = []
    extra_real = []

    for i in range(len(ROLES)):
        if brd[i] == "--":
            continue
        co = co_hist[i]
        if co is None:
            # Silent, and no wolf/Madman remains to be hiding
            # anywhere outside the claimed roles -- confirmed
            # villager.
            extra_villagers.append(i)
        elif claim_counts.get(co, 0) == 1:
            # The sole claimant of a role that needed 0 fakes,
            # with none left over -- confirmed genuine.
            extra_real.append(i)

    return extra_villagers, extra_real


def _describe_execution_choice(target):
    """
    Human-readable label for what TODAY's execution represents
    strategically -- "占いロラ" (forcing/rolling the Seer contest),
    "灰処刑" (no claim at all -- pure suspicion), etc. Purely
    DESCRIPTIVE: it just reads the same co_history/claim counts the
    execution decision itself already used, it isn't a new judgment
    or a plan of its own (the policy never reasons in these terms --
    this only translates its output back into them for the log).

    MUST be called once per execution, in order, BEFORE execute_player
    marks the target dead -- both for its own alive_claimants count
    below, and because it updates `_pending_roll` (module-level) for
    the NEXT call to read, which is how it tells a "ロラ" that
    continues into tomorrow from one that gets abandoned today (see
    the ロラ/ロラ完遂/決め打ち block below).
    """
    global _pending_roll

    own_co = co_history[target]
    role_jp = {"Ps": "霊能", "Hn": "狩人", "Sy": "占い"}

    # 2026-10 user request: "ロラ" means two CONSECUTIVE days'
    # executions both targeting the same still-contested role. Y
    # yesterday's call (see the bottom of this function) left behind
    # `_pending_roll` describing exactly who ELSE was still alive in
    # that contest at the moment yesterday's target was chosen. If
    # that's narrowed down to exactly one remaining rival by today,
    # today's execution either SETTLES that contest (the rival finally
    # gets executed too -- "ロラ完遂", no role name needed since it's
    # inherently a continuation of yesterday's already-named roll) or
    # ABANDONS it (something else gets executed instead, while the
    # rival is still alive and uncontested -- "{role}決め打ち": the
    # village settled for its current guess rather than following the
    # roll through to the end). Either way this is a ONE-DAY-AHEAD
    # check only -- the block at the end of this function always
    # recomputes `_pending_roll` fresh from TODAY's own execution, so
    # an abandoned contest doesn't keep echoing "決め打ち" on every
    # later day, only the single day right after it was dropped.
    override = None
    if _pending_roll is not None:
        rivals_still_alive = [i for i in _pending_roll["others"] if board[i] != "--"]
        if len(rivals_still_alive) == 1:
            rival = rivals_still_alive[0]
            if target == rival:
                override = "ロラ完遂"
            elif own_co == _pending_roll["role"]:
                # Only call this "決め打ち" (abandoning the roll) when
                # TODAY's execution is actually part of the SAME
                # contest the pending roll was tracking (own_co
                # matches the rolled role). 2026-10-04 bug fix: before
                # this check, executing a player who claimed a
                # DIFFERENT role entirely (e.g. a Hunter-CO contest
                # resolving on its own) would still inherit the old,
                # unrelated pending_roll's role name in the label --
                # e.g. a leftover Seer-contest pending_roll turning a
                # same-day Hunter-roll execution into the nonsensical
                # "占い決め打ち" instead of "狩人ロラ"/"狩人吊り". When
                # the roles don't match, leave override unset so the
                # normal own_co-based logic below picks the label.
                override = f"{role_jp[_pending_roll['role']]}決め打ち"

    if override is not None:
        label = override
    elif own_co not in village_ai.CLAIMABLE_ROLES:
        label = "村人CO対象の処刑" if own_co == "Vi" else "灰処刑(無CO対象)"
    else:
        # "ロラ(ローラー)" means forcing/narrowing a contest between 2+
        # STILL-LIVING same-role claimants. A claimant whose only
        # rival(s) already died isn't part of a live contest anymore,
        # even though the TOTAL claim count (dead rivals included) is
        # still >=2 -- so this counts only claimants who are ALIVE
        # right now, matching how the term is actually used (a lone
        # remaining claimant getting executed is just an ordinary
        # suspicion-based "◯◯吊り", not a roll).
        alive_claimants = sum(
            1 for i, co in enumerate(co_history)
            if co == own_co and board[i] != "--"
        )
        if alive_claimants >= 2:
            label = f"{role_jp[own_co]}ロラ"
        else:
            label = f"{role_jp[own_co]}吊り"

    # 2026-10-05 user request: "ロラ" must be CARRIED THROUGH to count.
    # (a) If yesterday's execution was labelled "◯◯ロラ" but today the
    # village did NOT execute the remaining rival (the rival is still
    # alive), yesterday was really a "決め打ち" -- fix yesterday's entry
    # retroactively (run_game rewrites the already-printed line from
    # _label_history after the game). (b) A game only has rope margin
    # for ONE real roll, so a second "◯◯ロラ" label in the same game is
    # shown as "決め打ち" instead. Purely descriptive -- nothing reads
    # these labels back into decisions.
    if _pending_roll is not None and _label_history:
        prev = _label_history[-1]
        alive_rivals = [i for i in _pending_roll["others"] if board[i] != "--"]
        if prev["is_roll"] and alive_rivals and target not in alive_rivals:
            prev["label"] = f"{role_jp[_pending_roll['role']]}決め打ち"
            prev["is_roll"] = False
    is_roll = label.endswith("ロラ") and label != "ロラ完遂"
    # A roll that CONTINUES yesterday's roll of the same role (e.g. 3
    # seer claimants: day1 ロラ, day2 ロラ, day3 ロラ完遂) is one roll,
    # not a "second" one. Only a NEW roll after an earlier one is
    # shown as 決め打ち.
    continues_roll = (
        is_roll and bool(_label_history)
        and _label_history[-1]["is_roll"]
        and _label_history[-1].get("role") == own_co
    )
    if is_roll and not continues_roll and any(e["is_roll"] for e in _label_history):
        label = label[:-2] + "決め打ち"
        is_roll = False
    _label_history.append({"label": label, "is_roll": is_roll, "role": own_co})

    # Recompute `_pending_roll` for TOMORROW's call, from TODAY's own
    # execution only -- any OLD pending roll (e.g. the one that just
    # got "決め打ち"-abandoned above, if that's what happened) is
    # always replaced here, never left lingering past the single day
    # right after it was left open.
    if own_co in village_ai.CLAIMABLE_ROLES:
        other_rivals = [
            i for i, co in enumerate(co_history)
            if co == own_co and board[i] != "--" and i != target
        ]
        _pending_roll = {"role": own_co, "others": other_rivals} if other_rivals else None
    else:
        _pending_roll = None

    return label


def _describe_attack_choice(target):
    """
    Wolf-side mirror of _describe_execution_choice: labels tonight's
    bite target the same way (e.g. "真占い噛み" -- attacking a Seer
    claimant the wolves have already narrowed down to genuinely real
    by elimination; NOT "白噛み" -- that term means biting someone
    the real Seer has already CLEARED as white, a different target
    entirely from the claimant themselves). Reuses the EXACT SAME
    "known real by elimination" logic wolf_ai._extract_attack_features
    itself scores on (is_known_real_seer/is_known_real_psychic), so
    this label always matches what the attack decision actually saw --
    including that "known real" here is the WOLVES' OWN private-
    knowledge elimination (claimant isn't a wolf teammate), which can,
    same as always, actually be a bluffing Madman rather than a true
    Seer/Medium -- this label doesn't know or claim otherwise.
    """
    own_co = co_history[target]
    wolf_indices = [i for i, role in enumerate(ROLES) if role == "Wo" and board[i] != "--"]

    sy_claimants = [i for i, co in enumerate(co_history) if co == "Sy"]
    known_real_seers = [i for i in sy_claimants if i not in wolf_indices]
    ps_claimants = [i for i, co in enumerate(co_history) if co == "Ps"]
    known_real_psychics = [i for i in ps_claimants if i not in wolf_indices]

    if target in known_real_seers:
        return "真占い噛み"
    if target in known_real_psychics:
        return "真霊能噛み"
    if own_co == "Hn":
        return "狩人噛み"
    if own_co == "Sy":
        return "対抗占いCO噛み"
    if own_co == "Ps":
        return "対抗霊能CO噛み"
    return "灰噛み"


def _current_day1_config():
    """
    (Sy, Ps, Hn) claim counts, counting ONLY claims made on Day 1
    (claim_day[i] == 1) -- the same computation train_and_report.py's
    day1_config_report does post-hoc, extracted here so LIVE in-game
    code (board_split.py's per-board-config routing) can call it too.

    Safe to call any time after day_phase(1) has finished: co_history/
    claim_day only ever get a role WRITTEN ONCE per player (nothing in
    this codebase ever overwrites an existing claim with a different
    one), so a player's claim_day == 1 status, once true, stays true
    for the rest of the game -- this doesn't need its own separate
    "freeze at end of day 1" snapshot/global the way some other
    per-game state does.

    NOT stable if called DURING day_phase(1) itself, though -- Day-1
    CO decisions are applied to co_history one at a time as they
    happen (medium -> hunter -> seer -> wolves -> madman), so a call
    partway through that sequence sees only however many of them have
    landed SO FAR, not the final count. This is why board_split.py's
    routing (via village_ai.execution_split / wolf_ai.attack_split)
    is safe: execution voting and the night attack both structurally
    happen only AFTER day_phase(1)'s entire CO sequence has finished
    (see day_phase's own call order) -- but anything that reads
    state["day1_config"] from an EARLIER Day-1 decision's logged state
    (e.g. for analysis/debugging) should not assume it's the final
    settled value.
    """
    sy = sum(1 for i, co in enumerate(co_history) if co == "Sy" and claim_day[i] == 1)
    ps = sum(1 for i, co in enumerate(co_history) if co == "Ps" and claim_day[i] == 1)
    hn = sum(1 for i, co in enumerate(co_history) if co == "Hn" and claim_day[i] == 1)
    return (sy, ps, hn)


def _public_guard_vouches(co_history_view):
    out = {}
    for claimant, targets in fake_declarations.items():
        out[claimant] = list(targets)
    real_idx = next((i for i, role in enumerate(ROLES) if role == "Hn"), None)
    if (real_idx is not None and co_history_view[real_idx] == "Hn"
            and guard_success_targets):
        out[real_idx] = list(guard_success_targets)
    # insertion order (fakes first, real last) must not be a tell
    return {k: out[k] for k in sorted(out)}


def _claimed_wolf_ids():
    """Players named a wolf by Medium claims, as a claim not a proof:
    a SOLE Ps claimant's black, or -- with 2+ claimants -- a player
    EVERY claimant who spoke on them called black (2026-10-06, H2;
    real and fake claimants look identical here)."""
    ps_claimants = [i for i, co in enumerate(co_history) if co == "Ps"]
    if not ps_claimants:
        return []
    out = []
    if len(ps_claimants) == 1:
        sole = ps_claimants[0]
        for r in medium_claims:
            if r["claimant"] == sole and r["is_wolf"] and r["player"] not in out:
                out.append(r["player"])
        return out
    by_player = {}
    for r in medium_claims:
        by_player.setdefault(r["player"], {})[r["claimant"]] = r["is_wolf"]
    for p, said in by_player.items():
        if len(said) >= 2 and all(said.values()) and p not in out:
            out.append(p)
    return out


def build_public_state(day, blind=False):
    """
    Parameters
    ----------
    blind : bool
        When True, every Day-1 CO claim is hidden from the returned
        state -- "board"/"co_history"/"claim_day" come back exactly
        as they looked BEFORE any Day-1 CO decision happened, no
        matter how many have already been made and applied to the
        real global board/co_history by the time this is called.

        This exists because Day-1 CO decisions are meant to be made
        SIMULTANEOUSLY (matching real Werewolf play: everyone
        declares at once, nobody has seen anyone else's claim yet),
        but they're actually processed one at a time in a fixed
        order (medium_day_co -> hunter_day_co -> seer_day_co ->
        wolf_day_co, and within wolf_day_co, wolf-by-wolf) with each
        claim applied to
        the shared board/co_history immediately. Without this, every
        decision after the first would see everything decided
        earlier THAT SAME DAY -- e.g. the Medium's Day-1 reveal is a
        forced, unconditional rule (see medium_day_co), so hunters
        and wolves would ALWAYS see revealed_Ps=True on Day 1,
        collapsing what should be an independent, blind decision
        into a purely reactive one.

        Only ever pass True for a Day-1 CO decision's own state; any
        OTHER decision (attack, vote, nomination, a CO decision on
        Day 2+) should see the real, unblinded state as normal.
    """

    if blind:
        board_view = ["Gr" if b != "--" else "--" for b in board]
        co_history_view = [None] * len(ROLES)
        claim_day_view = [None] * len(ROLES)
    else:
        board_view = list(board)
        co_history_view = list(co_history)
        claim_day_view = list(claim_day)

    real_hunter_index = next(
        (i for i, role in enumerate(ROLES) if role == "Hn"), None
    )
    num_hunter_claimants = sum(1 for co in co_history_view if co == "Hn")

    # Anyone who has ever actually DIED to a night attack is
    # unconditionally confirmed non-wolf (wolves never attack their
    # own) -- unlike the guard-success case below, this needs no
    # Hunter-identity gating at all, since there's no claimant's
    # self-report standing between the event and public knowledge: a
    # named player is simply dead the next morning, for everyone to
    # see, full stop.
    confirmed_villagers = list(attack_victims)

    # 2026-10-05 information-asymmetry fix: a Hunter's guard history is
    # trusted as "confirmed white" ONLY once that Hunter is proven real
    # by public logic -- guard-instruction verification
    # (confirmed_real_hunter) or the wolf-counting pigeonhole identity
    # proof (computed just below). The old rule also trusted a SOLE
    # Hunter claimant, but that trusted the real Hunter's true history
    # while ignoring a fake lone claimant's lie -- so the village could
    # tell them apart, which no human can.
    counting_villagers, counting_real = _wolf_counting_deduction(
        day, co_history_view, board_view
    )
    # 2026-10-06 (M6): "proven" is decided from PUBLIC proof only
    # (guard-instruction verification, or the pigeonhole identity
    # proof) and never by comparing against the true Hunter's index;
    # whatever claimant is publicly proven gets its own declared
    # history trusted (a real Hunter's true successes, or -- in the
    # rare case the public proof lands on a fake -- its declared lie,
    # exactly as a human village would).
    proven_hunters = []
    if confirmed_real_hunter is not None:
        proven_hunters.append(confirmed_real_hunter)
    for i in counting_real:
        if co_history_view[i] == "Hn" and i not in proven_hunters:
            proven_hunters.append(i)
    for h in proven_hunters:
        if co_history_view[h] != "Hn":
            continue
        # the claimant's PUBLIC vouch list (same shape for real/fake)
        for t in _public_guard_vouches(co_history_view).get(h, []):
            if t not in confirmed_villagers:
                confirmed_villagers.append(t)

    # A claimant PROVEN fake (via the same verification) whose
    # declaration falsely vouched for someone is itself a clue --
    # that declared target might be a wolf teammate the liar was
    # trying to protect.
    falsely_vouched_by_fake = [
        target
        for claimant in confirmed_fake_hunters
        for target in fake_declarations.get(claimant, [])
    ]

    # Global pigeonhole deduction across ALL claimed roles at once
    # (not just the Hunter-specific verification above): if the
    # claim counts exactly saturate the known total wolf count,
    # every uncontested claimant AND every silent player becomes
    # confirmed genuine with certainty.
    for i in counting_villagers:
        if i not in confirmed_villagers:
            confirmed_villagers.append(i)

    confirmed_real_hunter_final = confirmed_real_hunter
    confirmed_real_claimants = []
    for i in counting_real:
        if co_history_view[i] == "Hn" and confirmed_real_hunter_final is None:
            confirmed_real_hunter_final = i
        elif i not in confirmed_villagers:
            confirmed_villagers.append(i)
        confirmed_real_claimants.append(i)
    if confirmed_real_hunter_final is not None and confirmed_real_hunter_final not in confirmed_real_claimants:
        confirmed_real_claimants.append(confirmed_real_hunter_final)

    # NOTE: there used to be a "Seer cascade" here -- a Sy claimant
    # found among confirmed_villagers was treated as, with total
    # certainty, the genuine Seer (only wolves ever fake a role claim,
    # so a non-wolf Sy claimant had no other explanation), and their
    # reports were laundered straight into ground truth: a "not wolf"
    # verdict got appended to confirmed_villagers directly, and a
    # "wolf" verdict was surfaced as seer_confirmed_wolf_accusations
    # for village_ai's hard-rule execution override.
    #
    # That reasoning broke the moment the Madman was added: Madman is
    # confirmed non-wolf by exactly the same means (guard-instruction
    # verification, the wolf-counting pigeonhole, surviving a night
    # attack -- none of those check literal role) while being free to
    # ALSO bluff "Sy". Trusting such a claimant's reports as ground
    # truth here would have let a Madman's fabricated "not wolf"
    # verdict launder an actual wolf teammate straight into
    # confirmed_villagers -- a corruption of a list used pervasively
    # elsewhere as ground truth.
    #
    # 2026-10-04 REINSTATED, in a Madman-safe form: confirmed_real_
    # claimants above (built ONLY from the wolf-counting pigeonhole,
    # which explicitly accounts for the full total_wolves +
    # total_madmen fake-claim budget before declaring anyone real --
    # NOT from mere "not a wolf" membership in confirmed_villagers) is
    # an IDENTITY proof, not just a non-wolf proof. So a Sy claimant
    # found there specifically (as opposed to anywhere in the broader
    # confirmed_villagers) really can only be the one genuine Seer, no
    # Madman loophole possible -- their reports are safely laundered
    # below: a "white" verdict strengthens confirmed_villagers, and a
    # "black" verdict goes into the new confirmed_wolves list, both
    # with full certainty. (village_ai._extract_features separately
    # computes the weaker, ordinary-learned is_confirmed_contradicting_
    # seer / is_accused_by_confirmed_real_seer signals for the cases
    # that DON'T reach this bar -- see that module's 2026-10-04 notes.)
    confirmed_wolves = []
    for i in confirmed_real_claimants:
        if co_history_view[i] != "Sy":
            continue
        for r in seer_reports:
            if r["claimant"] != i:
                continue
            if r["is_wolf"]:
                if r["player"] not in confirmed_wolves:
                    confirmed_wolves.append(r["player"])
            else:
                if r["player"] not in confirmed_villagers:
                    confirmed_villagers.append(r["player"])

    # Information every side is allowed to see.
    return {
        "board": board_view,
        "co_history": co_history_view,
        "claim_day": claim_day_view,
        "day": day,
        "medium_reports": list(medium_reports),
        "medium_claims": list(medium_claims),
        "claimed_wolf_ids": _claimed_wolf_ids(),
        "guard_ever_succeeded": guard_ever_succeeded,
        "guard_success_count": guard_success_count,
        "confirmed_villagers": confirmed_villagers,
        "confirmed_fake_hunters": list(confirmed_fake_hunters),
        "confirmed_real_hunter": confirmed_real_hunter_final,
        "confirmed_real_claimants": confirmed_real_claimants,
        "confirmed_wolves": confirmed_wolves,
        "falsely_vouched_by_fake": falsely_vouched_by_fake,
        # 2026-10-05: ATTRIBUTED vouches of EVERY Hunter claimant, in one
        # symmetric shape -- a fake's lie target and a real Hunter's true
        # guard history look the same here. (The raw fake_declarations
        # dict used to be exposed instead, and its mere membership
        # identified the fake Hunters.)
        "guard_vouches": _public_guard_vouches(co_history_view),
        "guard_instructions": dict(guard_instructions),
        "nominations": nominations.copy(),
        "emergency_reveals": list(emergency_reveals),
        "emergency_wolf_claims": list(emergency_wolf_claims),
        "total_wolves": sum(1 for role in config.ROLE_POOL_10 if role == "Wo"),
        "confirmed_dead_wolves": len({
            p for p in ([r["player"] for r in medium_reports if r["is_wolf"]] + list(confirmed_wolves))
            if board[p] == "--"
        }),
        "execution_history": list(execution_history),
        "vote_history": {d: dict(v) for d, v in vote_history.items()},
        "suspicious_non_voters": dict(missed_wolf_vote_count),
        "defied_sole_nomination_count": dict(defied_nomination_count),
        "voted_for_protected_count": dict(voted_for_protected_count),
        "nominator_inconsistency": dict(nominator_inconsistency),
        "not_voting_own_black": dict(not_voting_own_black),
        "off_line_voters": dict(off_line_voters),
        "seer_reports": list(seer_reports),
        "self_declared_non_village": sorted(self_declared_non_village),
        "endgame_claims": {k: endgame_claims[k] for k in sorted(endgame_claims)},
        "alive_count": sum(1 for b in board_view if b != "--"),
        # Used only by board_split.py's per-Day-1-board-config routing
        # (village_ai.execution_split / wolf_ai.attack_split) -- see
        # _current_day1_config's own docstring for why this is safe to
        # compute live, unconditionally, regardless of `blind`.
        "day1_config": (None if blind else _current_day1_config()),
    }


def _teammate_indices(self_index, role):
    """
    Living players sharing `role` with `self_index`, excluding
    `self_index` itself -- e.g. the other living wolf(ves) for
    role="Wo". This is WOLF/MADMAN-PRIVATE knowledge (who your own
    teammates are), used to let wolf_ai._co_feature_list's new
    teammate_claimed_flags tell "my own partner already claimed this
    role" apart from the plain public "someone (possibly an enemy)
    already claimed this role" signal it already had. For role="Kg"
    this is normally empty (config.py's composition has exactly one
    Madman), which is the semantically correct "nobody to coordinate
    with" answer, not a special case that needs handling.
    """
    return [
        j for j, r in enumerate(ROLES)
        if r == role and board[j] != "--" and j != self_index
    ]


def build_wolf_state(day, targets=None):

    state = build_public_state(day)

    # Private knowledge: wolves know who the other wolves are.
    state["wolf_indices"] = [
        i for i, role in enumerate(ROLES)
        if role == "Wo" and board[i] != "--"
    ]

    # Public GJ history (the saved names are announced), used by the
    # wolf attack policy.
    state["guard_success_targets"] = list(guard_success_targets)

    if targets is not None:
        state["targets"] = targets

    return state


def build_village_state(day, candidates):

    state = build_public_state(day)

    # The village has no private information beyond the public
    # board -- only the set of players it is allowed to execute.
    state["candidates"] = candidates

    return state


def build_hunter_state(day):

    hunter_index = next(
        i for i, role in enumerate(ROLES)
        if role == "Hn" and board[i] != "--"
    )

    candidates = [
        i for i, s in enumerate(board)
        if s != "--" and i != hunter_index
    ]

    state = build_public_state(day)
    state["self_index"] = hunter_index
    state["candidates"] = candidates
    # PRIVATE to the hunter: last night's own guard choice.
    state["last_guard_target"] = last_guard_choice

    return state


def build_seer_state(day, self_index, candidates):

    state = build_public_state(day)
    state["self_index"] = self_index
    state["candidates"] = candidates
    # PRIVATE to the seer: only their own module ever reads this key,
    # and it is never included by any of the OTHER build_*_state
    # functions -- see seer_investigations' declaration for why.
    state["own_investigations"] = list(seer_investigations)
    # PUBLIC-information distribution of the village's vote (what any
    # player could estimate) -- 2026-10-06 seer_ai feature [8].
    state["village_vote_probs"] = village_ai.execution_distribution(
        build_village_state(day, [i for i, s in enumerate(board) if s != "--"]))

    return state


def log_decision(side, kind, state, action, day):
    """
    Record a single decision point (state seen -> action taken).

    This is the raw material a future training loop will use to
    assign rewards and compute policy updates. `state` is stored
    as given (already a fresh dict from build_*_state), so it is
    a snapshot, not a live reference.
    """
    history.append({
        "side": side,
        "type": kind,
        "day": day,
        "state": state,
        "action": action,
    })


# ============================================================
# Wolf CO (asked once per day, per still-unclaimed living wolf)
# ============================================================

_ENDGAME_CLAIM_TAG_JP = {"village": "村人CO", "wolf": "狼CO", "madman": "狂人CO"}


def _apply_endgame_claim(i, action, verbose=True):
    """
    Shared side effects for all three endgame identity-claim functions
    below (villager_endgame_bluff / madman_endgame_reveal /
    wolf_endgame_declare) -- all three offer the exact same four
    options (stay put, or claim_village/claim_wolf/claim_madman -- see
    wolf_ai.ENDGAME_DECLARE_OPTIONS's comment) and all three write into
    the exact same shared endgame_claims / self_declared_non_village
    public structures on a claim, so an outside observer (or a feature
    built from either structure) can never tell which of the three
    sides actually produced a given entry just by looking at it --
    that shared ambiguity is the whole point of the mechanic (see
    config.py's comment on "Kg", Steps 4-5).

    Kept intentionally terse (just the claimed identity, no flavor
    text) -- matching this session's move away from narrated "why" in
    every print line elsewhere (fake medium/seer reports, etc.) toward
    showing the bare public fact and letting deduction happen from the
    surrounding board state instead.
    """
    global self_declared_non_village, endgame_claims

    if action not in ("claim_village", "claim_wolf", "claim_madman"):
        return

    identity = action[len("claim_"):]
    endgame_claims[i] = identity
    if identity != "village":
        self_declared_non_village.append(i)

    if verbose:
        print(f"{PLAYERS[i]}が正体を明かした:【{_ENDGAME_CLAIM_TAG_JP[identity]}】")


def villager_endgame_bluff(day, verbose=True):
    """
    STEP 5 of the Madman addition (see config.py's comment on "Kg").

    Once only 3 or 4 players remain alive, each living, still-
    unclaimed plain Villager gets to decide, today, whether to falsely
    claim a non-village identity (or the subtler "村人CO") -- writing
    into the exact same endgame_claims / self_declared_non_village
    structures the Madman's genuine endgame reveal uses (see
    villager_ai.choose_endgame_bluff_action's docstring for why this
    specific ambiguity is the whole point). Restricted to villagers
    who haven't claimed anything else yet (co_history[i] is None) --
    a Villager already publicly claiming Ps/Hn/Sy has a different
    persona to protect and isn't the "plain, otherwise-invisible
    villager" this bluff is meant to hide.

    Called in the same place as madman_endgame_reveal, so both sides
    of this mechanic are settled before today's nominations/vote.
    """

    alive_count = sum(1 for b in board if b != "--")
    if alive_count not in (3, 4):
        return

    for i, role in enumerate(ROLES):

        # 2026-10-06 user request (information-asymmetry fix): this used
        # to admit only plain Villagers with NO claim, while wolves and
        # madmen could declare even after claiming Ps/Hn/Sy -- so any
        # claimant showing up in endgame_claims was necessarily a
        # wolf/madman (a real Sy/Ps/Hn could never be there). Every
        # village-side role, claimant or not, now gets the same option.
        if (
            role not in ("Vi", "Sy", "Ps", "Hn")
            or board[i] == "--"
            or i in endgame_claims
        ):
            continue

        state = build_public_state(day)
        state["self_index"] = i
        state["alive_count"] = alive_count

        action = villager_ai.choose_endgame_bluff_action(state)
        log_decision("villager", "endgame_bluff", state, action, day=day)

        _apply_endgame_claim(i, action, verbose=verbose)


def _add_fake_declaration(i, day, side, verbose=True):
    """One more lie from fake Hunter claimant `i`: "I guarded <declared>
    last night, white". Appends to fake_declarations[i] (a list)."""
    others = [
        j for j in range(len(ROLES))
        if board[j] != "--" and j != i
    ]
    if side == "wolf":
        teammates = [
            j for j, r in enumerate(ROLES)
            if r == "Wo" and board[j] != "--" and j != i
        ]
        strategy_state = build_public_state(day)
        _alive = sum(1 for b in board if b != "--")
        strategy_state["fake_decl_ctx"] = [
            1.0 if any(co_history[j] == "Hn" and j != i and board[j] != "--"
                       for j in range(len(ROLES))) else 0.0,
            1.0 if guard_success_count > 0 else 0.0,
            1.0 if teammates else 0.0,
            day / 7.0,
            1.0 if _alive <= 5 else 0.0,
            1.0 if any(t in teammates or ROLES[t] == "Wo"
                       for t in fake_declarations.get(i, [])) else 0.0,
        ]
        strategy = wolf_ai.choose_fake_declaration_strategy(strategy_state)
        log_decision("wolf", "fake_declaration_strategy",
                     strategy_state, strategy, day=day)
        if strategy == "declare_teammate" and teammates:
            declared = random.choice(teammates)
        elif others:
            declared = random.choice(others)
        else:
            declared = None
    else:
        # The Madman has no teammate to protect, so a random living
        # player is the only meaningful choice.
        declared = random.choice(others) if others else None
    if declared is not None:
        fake_declarations.setdefault(i, []).append(declared)
        if verbose:
            print(f"(嘘)「{PLAYERS[declared]} を護衛しました、"
                  f"白です」と{PLAYERS[i]}が主張(※本当は偽り)")


def _extend_fake_declarations(day, verbose=True):
    """Called right after a guard success (a peaceful night): every
    living FAKE Hunter claimant adds one more vouch, so its public
    history keeps the same length as a real Hunter's."""
    for i, role in enumerate(ROLES):
        if board[i] == "--" or co_history[i] != "Hn" or role not in ("Wo", "Kg"):
            continue
        _add_fake_declaration(i, day, "wolf" if role == "Wo" else "madman", verbose=verbose)


def _apply_fake_claim(i, day, action, side, verbose=True):
    """
    Apply the board/co_history side effects of a FAKE claimant's CO
    decision (an action from wolf_ai.choose_co_action or madman_ai.
    choose_co_action: "claim_<role>" or "stay_silent"). `side` is
    "wolf" or "madman" -- both share this exact same side-effect
    logic (claim + announce + the fake-Hunter vouching lie), only the
    STRATEGY behind who gets vouched for differs by side (see below).

    Factored out of wolf_day_co/madman_day_co so the exact same side
    effects apply whether this claim came from the routine daily
    check or from the last-second emergency evasion check in
    progression_ai -- a threatened bluffer reaching for a desperate
    fake claim should behave identically to one made on a normal day,
    just triggered at a different, more urgent moment.

    Returns
    -------
    bool
        True if a claim was actually made (board/co_history changed),
        False if this player chose to stay silent.
    """

    if action == "stay_silent":
        return False

    # action is "claim_<role_code>" for one of the configured
    # CLAIMABLE_ROLES, e.g. "claim_Ps".
    claimed_role = action[len("claim_"):]
    board[i] = claimed_role
    co_history[i] = claimed_role
    claim_day[i] = day
    announce_co(i, claimed_role, verbose=verbose, day=day)

    if claimed_role == "Hn" and day > 1:
        # A fake Hunter claim: also choose someone to falsely vouch
        # for as "confirmed white". This is a lie with no real
        # backing -- it only matters if the village mistakenly
        # trusts it before verification resolves who's genuine.
        #
        # Only possible from Day 2 onward: a guard action only
        # happens at NIGHT, so on Day 1 itself nobody -- real or
        # fake -- has any guard history to claim yet. Claiming one
        # on Day 1 would be an instant, structural giveaway
        # regardless of who's genuine.
        # 2026-10-06 (M2, information-asymmetry fix): a real Hunter's
        # public history has one entry per PEACEFUL night (a guard
        # success), so a fake must show the same count -- one declared
        # vouch per peaceful night so far (0 on Day 1 / before any
        # peaceful night, i.e. no entry at all), and it keeps adding
        # one on every later peaceful night (see
        # _extend_fake_declarations). Before, a fake always had exactly
        # one entry from the moment of its claim.
        for _ in range(len(guard_success_targets)):
            _add_fake_declaration(i, day, side, verbose=verbose)

    return True


def wolf_day1_joint_co(day, verbose=True):
    """
    DAY 1 ONLY. The two wolves decide their Day-1 claims TOGETHER, as
    a single pre-agreed plan, instead of each independently drawing
    from wolf_ai.co_policy the way wolf_day_co (below, Day 2+) still
    does -- see wolf_ai.py's module comment above
    choose_day1_joint_co_action for the full rationale. In short: a
    real wolf pair can agree on a joint plan overnight even though
    they can't react to each other DURING the simultaneous morning
    announcement itself, and the old independent-per-wolf approach
    had no way to represent that (its teammate-awareness feature was
    PROVABLY stuck at zero gradient on Day 1 specifically -- see
    wolf_ai.py for the proof). This models it directly as one joint
    choice, applied to both wolves at once.

    Only ever runs with exactly 2 living wolves in practice (this
    composition's fixed wolf count, config.ROLES -- and nobody can
    die before Day 1 in this game's flow to begin with). The `else`
    branch below is a defensive fallback for a hypothetical future
    composition change, not a path expected to ever actually trigger;
    it just reuses the ordinary independent policy exactly as Day 2+
    does, rather than crashing or silently doing nothing.
    """

    wolves = [
        i for i, role in enumerate(ROLES)
        if role == "Wo" and board[i] != "--" and co_history[i] is None
    ]

    if len(wolves) != 2:
        for i in wolves:
            state = build_public_state(day, blind=True)
            state["teammate_indices"] = _teammate_indices(i, "Wo")
            action = wolf_ai.choose_co_action(state)
            log_decision("wolf", "co_action", state, action, day=day)
            _apply_fake_claim(i, day, action, side="wolf", verbose=verbose)
        return

    action = wolf_ai.choose_day1_joint_co_action()
    log_decision("wolf", "day1_joint_co", build_public_state(day), action, day=day)

    # The chosen pair is UNORDERED (see wolf_ai.py) -- which physical
    # wolf gets which half is decided uniformly at random here, so
    # the model is never pushed to learn a meaningless "player index"
    # preference that has nothing to do with actual strategy.
    pair = list(action)
    random.shuffle(pair)
    _apply_fake_claim(wolves[0], day, pair[0], side="wolf", verbose=verbose)
    _apply_fake_claim(wolves[1], day, pair[1], side="wolf", verbose=verbose)


def wolf_day_co(day, verbose=True):
    """
    Each living wolf who hasn't claimed anything yet independently
    decides, today, whether to claim Medium, claim Hunter, claim
    Seer, or stay silent. Called AFTER medium_day_co/hunter_day_co/
    seer_day_co in day_phase, so wolves can react to whatever was
    just revealed today.

    DAY 1 IS HANDLED SEPARATELY (see wolf_day1_joint_co above) -- this
    function is a no-op on Day 1 and only ever actually runs from
    Day 2 onward, where each wolf's decision is genuinely sequential
    and reactive rather than a simultaneous joint announcement.
    """

    if day == 1:
        return

    for i, role in enumerate(ROLES):

        if role == "Wo" and board[i] != "--" and co_history[i] is None:

            state = build_public_state(day, blind=False)
            state["teammate_indices"] = _teammate_indices(i, "Wo")
            action = wolf_ai.choose_co_action(state)
            log_decision("wolf", "co_action", state, action, day=day)

            _apply_fake_claim(i, day, action, side="wolf", verbose=verbose)


def madman_day_co(day, verbose=True):
    """
    Each living Madman who hasn't claimed anything yet independently
    decides, today, whether to claim Medium, Hunter, or Seer, or stay
    silent -- Step 3 of the Madman addition (see config.py's comment
    on "Kg", and madman_ai's module docstring for why this reuses
    wolf_ai's CO decision machinery wholesale). Called AFTER
    wolf_day_co, so the Madman -- who has no teammate to coordinate
    with and therefore nothing to lose by acting last -- can react to
    whatever the real roles AND the wolves themselves just revealed
    today, exactly like a wolf reacting to a same-day Hunter/Seer
    reveal already could.

    There is normally at most one Madman in this composition
    (config.ROLES), but this loops over every "Kg" the same way
    wolf_day_co loops over every "Wo", in case that ever changes.
    """

    for i, role in enumerate(ROLES):

        if role == "Kg" and board[i] != "--" and co_history[i] is None:

            state = build_public_state(day, blind=(day == 1))
            state["teammate_indices"] = _teammate_indices(i, "Kg")
            action = madman_ai.choose_co_action(state)
            log_decision("madman", "co_action", state, action, day=day)

            _apply_fake_claim(i, day, action, side="madman", verbose=verbose)


def madman_endgame_reveal(day, verbose=True):
    """
    STEP 4 of the Madman addition (see config.py's comment on "Kg").

    Once only 3 or 4 players remain alive, each living Madman who
    hasn't already revealed gets to decide, today, whether to openly
    declare itself hostile to the village -- see madman_ai's "STEP 4"
    section for the full mechanic and why this can help the wolves'
    vote converge on the real villager instead of accidentally
    splitting among all the remaining ambiguous candidates.

    Unlike madman_day_co (which only asks while co_history[i] is
    None), this is asked regardless of whether the Madman already has
    an active Ps/Hn/Sy claim from Step 3 -- revealing means abandoning
    that cover too, which is exactly the already_has_role_claim
    feature's job to weigh.

    Called BEFORE make_all_nominations/the day's vote (see day_phase),
    so a reveal made today can actually influence today's outcome --
    an endgame reveal that only became public AFTER the vote would be
    strategically useless.

    There is normally at most one Madman in this composition, but this
    loops over every "Kg" the same way madman_day_co does, in case
    that ever changes.
    """

    alive_count = sum(1 for b in board if b != "--")
    if alive_count not in (3, 4):
        return

    for i, role in enumerate(ROLES):

        if role != "Kg" or board[i] == "--" or i in endgame_claims:
            continue

        state = build_public_state(day)
        state["self_index"] = i
        state["alive_count"] = alive_count

        action = madman_ai.choose_endgame_reveal_action(state)
        log_decision("madman", "endgame_reveal", state, action, day=day)

        _apply_endgame_claim(i, action, verbose=verbose)


def wolf_endgame_declare(day, verbose=True):
    """
    Symmetric counterpart to madman_endgame_reveal, added after a
    real played-out game showed the structural hole it left open: only
    a genuine Madman or a genuine (bluffing) Villager could ever end up
    in self_declared_non_village, so once every OTHER living player had
    declared, the one holdout was trivially identifiable as the wolf --
    no suspicion feature needed, just elimination by counting. Giving a
    living wolf this exact same option (wolf_ai.choose_endgame_declare_
    action) closes that hole: a wolf can now ALSO show up in self_
    declared_non_village, so "did this player declare?" stops being a
    wolf-identifying signal on its own for anyone. See wolf_ai's
    ENDGAME_DECLARE_OPTIONS section for the full reasoning on why a
    wolf would ever want this (pure self-defense, not coordination).

    Same alive_count in (3, 4) gate, same "already declared" exclusion,
    and same shared endgame_claims / self_declared_non_village
    structures as madman_endgame_reveal -- called right alongside it,
    in the same phase, so a wolf's decision can react to (or be
    reacted to by) the SAME day's Madman/Villager declarations before
    the day's vote.
    """

    alive_count = sum(1 for b in board if b != "--")
    if alive_count not in (3, 4):
        return

    for i, role in enumerate(ROLES):

        if role != "Wo" or board[i] == "--" or i in endgame_claims:
            continue

        state = build_public_state(day)
        state["self_index"] = i
        state["alive_count"] = alive_count

        action = wolf_ai.choose_endgame_declare_action(state)
        log_decision("wolf", "endgame_declare", state, action, day=day)

        _apply_endgame_claim(i, action, verbose=verbose)


# ============================================================
# Role Candidates
# ============================================================

def get_role_candidates():

    candidates = []

    for i in range(len(PLAYERS)):

        if co_history[i] in CLAIMABLE_ROLES:
            # Wolves, the Madman, and (in principle) a plain Villager
            # can all fake this claim now (Step 3: the Madman can
            # bluff Ps/Hn/Sy just like a wolf can -- see
            # madman_day_co).
            candidates.append(["Wo", "Vi", "Kg", co_history[i]])
        elif co_history[i] == "Vi":
            # Experimental: a wolf claiming to be a plain Villager
            # (something real human play generally treats as
            # pointless/prohibited). Kept as an option to see
            # whether self-play independently learns to avoid it.
            candidates.append(["Wo", "Vi"])
        else:
            # Any claim-bound role may still be silently unclaimed.
            candidates.append(["Wo", "Vi", "Kg"] + list(CLAIMABLE_ROLES))

    return candidates


# ============================================================
# Progression AI (village execution vote)
# ============================================================

def progression_ai(day, verbose=True):

    candidates = [
        i for i, state in enumerate(board)
        if state != "--"
    ]

    state = build_village_state(day, candidates)
    # 2026-10-06 (L4): the "threat" signal is now the MOST LIKELY target
    # of the village's public-information vote distribution (what any
    # human could estimate), not one random sampled draw of it.
    _threat_dist = village_ai.execution_distribution(state)
    preview_target = max(candidates, key=lambda c: (_threat_dist.get(c, 0.0), -c))

    # If the village's leading target is the still-hidden real
    # Hunter, give them ONE emergency chance to reveal and save
    # themselves -- a different decision point from the routine
    # daily check in hunter_day_co, since this one specifically
    # knows "you are about to be executed right now".
    true_hunter_idx = next(
        (i for i, role in enumerate(ROLES) if role == "Hn" and board[i] != "--"),
        None
    )

    emergency_triggered = (
        true_hunter_idx is not None
        and co_history[true_hunter_idx] is None
        and (preview_target == true_hunter_idx or true_hunter_idx in nominations.values())
    )

    if emergency_triggered:

        threat_state = build_public_state(day)
        threat_state["gray_count"] = sum(
            1 for j, s in enumerate(board)
            if s != "--" and co_history[j] is None
        )
        threat_state["under_threat"] = True
        threat_state["personally_nominated"] = (true_hunter_idx in nominations.values())

        action = hunter_ai.choose_co_action(threat_state)
        log_decision("hunter", "co_action", threat_state, action, day=day)

        if action == "reveal":

            board[true_hunter_idx] = "Hn"
            co_history[true_hunter_idx] = "Hn"
            claim_day[true_hunter_idx] = day
            emergency_reveals.append(true_hunter_idx)

            if verbose:
                print(f"{PLAYERS[true_hunter_idx]} が処刑対象に指名され、緊急CO")
            announce_co(true_hunter_idx, "Hn", verbose=verbose)
            if verbose and guard_success_targets:
                saved = "、".join(PLAYERS[t] for t in guard_success_targets)
                print(f"   → これまでの護衛先を公表: {saved} は確定で白!")

            candidates = [i for i, s in enumerate(board) if s != "--"]

    # NOTE: there used to be a mirrored emergency-evasion branch here
    # for the Seer, exactly like the Hunter's above. It's gone now
    # that the Seer is ALSO forced to reveal Day 1 (seer_day_co, same
    # as medium_day_co): by the time progression_ai ever runs, the
    # real Seer's co_history entry can no longer be None (day_phase
    # always calls seer_day_co before progression_ai -- see its call
    # order), so the "still-hidden real Seer" this branch used to
    # react to can never exist. A wolf can still choose to fake-CO
    # "Sy" under threat, in the loop just below -- that's a completely
    # separate decision (wolf_ai's own CO policy, not the real Seer's).
    any_emergency_this_round = bool(emergency_triggered and action == "reveal")
    claimable_emergency_happened = any_emergency_this_round
    new_hunter_contest_this_round = False

    # Give each still-silent, living WOLF *or* MADMAN (Step 3: the
    # Madman can bluff too, and has exactly the same nothing-to-lose
    # incentive under threat that a wolf does) an immediate reaction
    # opportunity, THIS SAME DAY, in either of two situations:
    #
    #   1. This player is itself about to be executed (today's leading
    #      target, or personally named in the nomination) -- a
    #      last-second evasion gamble with nothing to lose, worst
    #      case identical to staying silent.
    #
    #   2. The real Hunter JUST emergency-revealed above (the Seer's
    #      equivalent branch is gone now that they're forced to
    #      reveal Day 1 -- see the NOTE above). A routine day's
    #      wolf_day_co/madman_day_co already ran and finished BEFORE
    #      this point (see day_phase's call order), so without this,
    #      a bluffer could only react to a same-day reveal starting
    #      the NEXT day -- by which point the "someone just revealed,
    #      contest now or don't" moment has already passed and a late
    #      fake claim looks obviously reactive. Reacting HERE,
    #      immediately, lets a bluffer contest the claim right when it
    #      would actually matter: for a Hunter claim, doing so
    #      triggers assign_guard_instructions' 2+-claimant
    #      verification that same night. Either way, no further
    #      deliberation is needed if nobody contests: the nominated
    #      player just gets executed as called.
    #
    # Both cases reuse each side's existing CO policy/options (wolf_
    # ai's for a wolf, madman_ai's for the Madman -- see wolf_day_co/
    # madman_day_co) -- case 2 doesn't even need a new context
    # feature, since the "did an emergency reveal just happen" flag
    # already feeds into _co_feature_list (previously only checked
    # again the NEXT day; state built here already reflects today's
    # reveal).
    for i, role in enumerate(ROLES):

        if role not in ("Wo", "Kg") or board[i] == "--" or co_history[i] is not None:
            continue

        side = "wolf" if role == "Wo" else "madman"
        module = wolf_ai if role == "Wo" else madman_ai

        personally_under_threat = (i == preview_target or i in nominations.values())
        if not (personally_under_threat or claimable_emergency_happened):
            continue

        threat_state = build_public_state(day)
        threat_state["under_threat"] = personally_under_threat
        threat_state["personally_nominated"] = (i in nominations.values())
        threat_state["teammate_indices"] = _teammate_indices(i, role)

        bluff_action = module.choose_co_action(threat_state)
        log_decision(side, "co_action", threat_state, bluff_action, day=day)

        if verbose and bluff_action != "stay_silent":
            if personally_under_threat:
                print(f"{PLAYERS[i]} が処刑対象に指名され、回避のため緊急CO")
            else:
                print(f"{PLAYERS[i]} が対抗して緊急CO")

        if _apply_fake_claim(i, day, bluff_action, side=side, verbose=verbose):
            any_emergency_this_round = True
            if co_history[i] in ("Ps", "Sy"):
                # This "Ps"/"Sy" claim is coming after that day's CO
                # phase AND its nomination already happened -- unlike
                # a routine same-day fake claim (ambiguous
                # "simultaneous morning announcement" timing), this is
                # unambiguously LATER than the real Medium's/Seer's
                # mandatory Day-1 reveal, which always happens first
                # (both roles are forced -- see medium_day_co and
                # seer_day_co). Recording it here lets village_ai.
                # _is_impossible_late_forced_claim treat it as a
                # 100%-certain fake claim, not just a suspicious one
                # -- see that function's docstring.
                emergency_wolf_claims.append(i)
            elif co_history[i] == "Hn":
                new_hunter_contest_this_round = True
                # 2026-10-05 asymmetry fix: emergency_reveals used to list
                # ONLY the real Hunter's emergency CO, so a wolf reading it
                # could tell a real emergency Hunter from a Madman's bluff.
                # Any emergency Hn claim now counts, real or fake.
                emergency_reveals.append(i)

    # assign_guard_instructions already ran earlier in day_phase, at
    # a point BEFORE any of the reactions above could exist yet -- so
    # if one of them just created a fresh 2-Hunter-claimant contest
    # (exactly the point of the "call for a counter-claim" reaction
    # above), re-run it now so tonight's guard-instruction
    # verification actually gets set up THIS day rather than being
    # silently missed because the claimant count changed too late.
    if new_hunter_contest_this_round:
        assign_guard_instructions(day, verbose=verbose)

    # Any emergency reveal just now (Hunter's or a wolf's evasion
    # claim) means an UNCONTESTED nominator's earlier call today was
    # made on STALE information -- give them one more chance to
    # re-target now that this new development is public, exactly as
    # a real player would reconsider who to point at. This overwrites
    # nominations[...] in place, so run_vote's defied_sole_nomination_
    # count check right below judges voters against the UPDATED call,
    # not the one made before the reveal.
    if any_emergency_this_round:
        for nominator, old_target in list(nominations.items()):

            if board[nominator] == "--":
                continue  # nominator isn't around to revise anything

            revise_candidates = [
                j for j in range(len(ROLES))
                if board[j] != "--" and j != nominator
            ]
            if not revise_candidates:
                continue

            revise_state = build_public_state(day)
            revise_state["voter"] = nominator
            if ROLES[nominator] == "Ps":
                revise_state["voter_private_truth"] = {
                    t: (ROLES[t] == "Wo") for t in execution_history if t is not None}

            # There used to be a skip here for when the situation was
            # ALREADY conclusively settled by a hard proof (e.g. a
            # structurally-impossible late Medium claim) -- the
            # execution vote's own hard-rule override guaranteed the
            # outcome regardless of what today's nomination said, so
            # re-asking the nomination policy would just reconsider a
            # settled fact as a live judgment call. That override is
            # gone now (see village_ai.choose_execution_target's
            # docstring -- a proven-fake claimant might be a Madman,
            # whose removal isn't unconditionally good for the
            # village the way a wolf's is), so the nomination is
            # always genuinely worth reconsidering here.
            new_target = medium_ai.choose_nomination_target(revise_state, revise_candidates)
            if ROLES[nominator] == "Ps":  # a fake medium's re-nomination is not trained (see _make_nomination)
                log_decision("medium", "nomination_target", revise_state, new_target, day=day)

            if new_target != old_target:
                nominations[nominator] = new_target
                if verbose:
                    print(f"   → {PLAYERS[nominator]}(指定者)が指定を変更: "
                          f"{PLAYERS[old_target]} → {PLAYERS[new_target]}")

    # Execution is no longer decided by fiat from a single "village
    # consensus" call -- `preview_target` above exists ONLY as a
    # cheap threat signal for the emergency-reveal checks just above.
    # The actual outcome now comes from a real vote: every living
    # player, wolves included, independently picks someone, and the
    # plurality gets executed (see run_vote).
    return run_vote(day, candidates, verbose=verbose)


def run_vote(day, candidates, verbose=True):
    """
    Every living player independently votes for one OTHER living
    player; whoever gets the most votes is executed (ties broken
    uniformly at random).

    Village-side voters (villager/medium/hunter alike) all share
    ONE policy -- village_ai.execution_policy -- since they all
    reason from the exact same honest, public information; each one
    just independently samples their own vote from it (the same
    pattern already used for the two wolves' shared CO/attack
    policies). Wolves use their OWN policy (wolf_ai.vote_policy),
    since their motive is completely different -- blend in, and
    above all never expose a teammate -- and they have private
    information (who their teammate is) no villager could have.
    """

    votes = {}

    for voter in candidates:

        vote_candidates = [c for c in candidates if c != voter]
        if not vote_candidates:
            continue

        if ROLES[voter] == "Wo":
            vstate = build_wolf_state(day)
            vstate["voter"] = voter
            vstate["vote_candidates"] = vote_candidates
            # The village's likely vote, computed from the SAME public
            # information a villager would use (see wolf_ai's
            # village_vote_prob vote feature).
            _vv_state = build_village_state(day, vote_candidates)
            vstate["village_vote_probs"] = village_ai.execution_distribution(_vv_state)
            target = wolf_ai.choose_vote_target(vstate)
            log_decision("wolf", "vote_target", vstate, target, day=day)
        elif ROLES[voter] == "Kg":
            # The Madman sees the exact same public information a
            # genuine villager does (no private wolf-knowledge --
            # confirmed design decision), so it reuses
            # build_village_state wholesale -- see madman_ai.py's
            # module docstring for why sharing the FEATURES while
            # keeping a completely separate policy/reward is the
            # right split here. Always uses its own free (logged)
            # choice, unlike a village-side voter's shared policy --
            # the Madman's incentive is the opposite of a village
            # voter's, so there was never anything to share beyond the
            # feature vector itself.
            vstate = build_village_state(day, vote_candidates)
            vstate["voter"] = voter
            target = madman_ai.choose_vote_target(vstate)
            log_decision("madman", "vote_target", vstate, target, day=day)
        else:
            vstate = build_village_state(day, vote_candidates)
            # Lets village_ai's is_this_voters_own_nomination feature
            # tell whether THIS SPECIFIC voter is the Medium claimant
            # who nominated a given candidate -- otherwise a nomination
            # and its nominator's own vote are two independent draws
            # from the same shared policy, with nothing tying them
            # together at all.
            vstate["voter"] = voter
            # 2026-10-06: the REAL Seer's own private results are part
            # of their own view when they vote (nobody else gets this).
            if ROLES[voter] == "Ps":
                # the real Medium privately KNOWS every executed player's alignment
                vstate["voter_private_truth"] = {
                    t: (ROLES[t] == "Wo") for t in execution_history if t is not None}
            if ROLES[voter] == "Sy":
                vstate["voter_private_truth"] = {
                    inv["target"]: inv["is_wolf"] for inv in seer_investigations}
                vstate["voter_private_whites"] = [
                    inv["target"] for inv in seer_investigations if not inv["is_wolf"]]
                vstate["voter_private_blacks"] = [
                    inv["target"] for inv in seer_investigations if inv["is_wolf"]]
            target = village_ai.choose_execution_target(vstate)
            # Every village vote is now this policy's genuine free
            # choice (village_ai.choose_execution_target no longer has
            # a hard-rule override -- see its docstring), so every one
            # of them is worth logging and training on.
            log_decision("village", "execution", vstate, target, day=day)

        votes[voter] = target

    vote_history[day] = votes

    # A sole (currently uncontested) Medium claimant's nomination is
    # trustworthy the moment it's made -- no hindsight needed, unlike
    # missed_wolf_vote_count above. Anyone who voted for someone else
    # today, while such a nomination stood against a still-living
    # target, gets counted here (see village_ai's
    # defied_sole_nomination_count feature docstring for why).
    claim_counts_today = {
        role: sum(1 for co in co_history if co == role) for role in CLAIMABLE_ROLES
    }
    for nominator, target in nominations.items():
        nominator_co = co_history[nominator]
        if nominator_co not in CLAIMABLE_ROLES:
            continue
        if claim_counts_today.get(nominator_co, 0) > 1:
            continue  # contested -- not a trustworthy nomination (yet)
        if board[target] == "--":
            continue  # nominee already dead -- nothing left to defy today
        if votes.get(nominator) is not None and votes.get(nominator) != target:
            nominator_inconsistency[nominator] = nominator_inconsistency.get(nominator, 0) + 1
        for voter, voted_target in votes.items():
            if voter == target or voter == nominator:
                continue
            if voted_target != target:
                defied_nomination_count[voter] = defied_nomination_count.get(voter, 0) + 1

    # 2026-10-06 (M5): a Sy claimant who voted for none of their own
    # (still-living) black targets.
    for _s, _co in enumerate(co_history):
        if _co != "Sy" or _s not in votes:
            continue
        _blacks = {r["player"] for r in seer_reports
                   if r["claimant"] == _s and r["is_wolf"] and board[r["player"]] != "--"
                   and r["player"] != _s}
        if _blacks and votes[_s] not in _blacks:
            not_voting_own_black[_s] = not_voting_own_black.get(_s, 0) + 1

    # 2026-10-05 user request: "方針外投票". The day's village line is
    # the set of ALL living claimants of the contested role a roll
    # would work through (Sy first, then Ps, then Hn; a role with fewer
    # than 2 living claimants isn't a roll). Anyone who voted for none
    # of them is counted -- see village_ai.off_line_vote_count.
    line_set = []
    for role in ("Sy", "Ps", "Hn"):
        claimants = [i for i, co in enumerate(co_history)
                     if co == role and board[i] != "--"]
        if len(claimants) >= 2:
            line_set = claimants
            break
    if line_set:
        for voter, voted_target in votes.items():
            if voted_target not in line_set:
                off_line_voters[voter] = off_line_voters.get(voter, 0) + 1

    # Mirror image of the check above: a vote FOR someone
    # village_ai.apply_protection_floor would have excluded (a sole/
    # uncontested claimant, an already-confirmed villager, or the
    # confirmed real Hunter) is something a genuine village-aligned
    # voter has no reason to do -- their own execution policy already
    # keeps such candidates out of its pool. Wolves have no equivalent
    # restraint in wolf_ai.vote_policy, so this flags exactly the kind
    # of vote a wolf occasionally casts (see voted_for_protected_count's
    # docstring) that a real villager essentially never would.
    protection_check_state = build_public_state(day)
    for voter, voted_target in votes.items():
        if village_ai._is_structurally_protected(protection_check_state, voted_target):
            voted_for_protected_count[voter] = voted_for_protected_count.get(voter, 0) + 1

    if not votes:
        # No living player had anyone else to vote for (shouldn't
        # normally happen with 2+ alive) -- fall back to a random
        # candidate rather than crashing.
        return random.choice(candidates)

    tally = {}
    for target in votes.values():
        tally[target] = tally.get(target, 0) + 1

    top_count = max(tally.values())
    top = [p for p, c in tally.items() if c == top_count]
    chosen = random.choice(top)

    if verbose:
        vote_str = "  ".join(f"{PLAYERS[v]}→{PLAYERS[t]}" for v, t in votes.items())
        print(f"【投票】{vote_str}")
        if len(top) > 1:
            tie_str = "・".join(PLAYERS[p] for p in top)
            print(f"   → {top_count}票で同数({tie_str}) → ランダムで{PLAYERS[chosen]}に決定")

    return chosen


# ============================================================
# Execution
# ============================================================

def execute_player(target):
    global execution_history
    board[target] = "--"
    execution_history.append(target)


# ============================================================
# Game End Check
# ============================================================

def check_game_end():

    alive_roles = [
        ROLES[i]
        for i, state in enumerate(board)
        if state != "--"
    ]

    wolves = alive_roles.count("Wo")
    village = len(alive_roles) - wolves

    if wolves == 0:
        return "Village"

    if wolves >= village:
        return "Werewolf"

    return None


# ============================================================
# Day Phase
# ============================================================

def assign_guard_instructions(day, verbose=True):
    """
    When 2+ Hunter claimants are alive, publicly assign each
    UNRESOLVED claimant a DIFFERENT guard target. Only the real
    Hunter's guard actually blocks an attack, so whichever
    claimant's assigned target dies tonight is thereby proven fake
    -- and whichever claimant's assigned target survives an attack
    is thereby proven genuine.

    2026-10-03 user request: up to 2 wolves AND the Madman can each
    independently fake-claim "Hn" on top of the one real Hunter, so 3
    simultaneous unresolved claimants is a real, reachable case (not
    just the common 2-claimant one). This now watches up to 3 of
    them -- using village_ai's separate, freshly-initialized 3-slot
    policy (choose_guard_instruction_strategy_3, 64 options) when
    there are 3+, and the original, already-trained 2-slot policy
    (16 options) when there are exactly 2, so the common case's
    learned weights are undisturbed.

    Which claimants get watched is now a RANDOM sample rather than
    "the `n` lowest player numbers" -- the old `unresolved[:2]` had
    no strategic meaning and simply always favored low seat numbers
    (e.g. with 3 claimants P5/P6/P7, P7 could never be watched until
    P5 or P6 resolved first, purely because of player numbering).

    2026-10-04 user request: a SOLE/uncontested claimant (exactly 1
    unresolved "Hn" claim, no rival to cross-check against) now also
    gets a guard instruction assigned and verified, via its own
    separate 1-slot policy (village_ai.choose_guard_instruction_
    strategy_1) -- previously this function returned early with no
    instruction at all whenever fewer than 2 claimants were unresolved,
    leaving a lone fake claim completely unverifiable.
    """

    global guard_instructions

    claimants = [
        i for i, co in enumerate(co_history)
        if co == "Hn" and board[i] != "--"
    ]

    unresolved = [
        c for c in claimants
        if c not in confirmed_fake_hunters and c != confirmed_real_hunter
    ]

    if len(unresolved) < 1:
        guard_instructions = {}
        return

    n_slots = min(len(unresolved), 3)
    watched = random.sample(unresolved, n_slots)

    candidates = [
        i for i in range(len(ROLES))
        if board[i] != "--" and i not in watched
    ]

    if len(candidates) < n_slots:
        guard_instructions = {}
        return

    # Let the village decide HOW to assign the watch slots -- each
    # slot independently targets a random gray player, the Medium,
    # the Seer, or (with 2+ claimants) one of the OTHER claimants.
    #
    # 2026-10-04 user request: this used to require 2+ unresolved
    # claimants before assigning ANY instruction at all, leaving a
    # single/uncontested "Hn" claim with zero of this mechanism's
    # fake/genuine verification -- see village_ai.py's
    # GUARD_INSTRUCTION_OPTIONS_1 module comment for the diagnosis
    # (this was the single largest underperforming board bucket in
    # the project's own game log database). n_slots == 1 now uses its
    # own separate, freshly-initialized policy (choose_guard_
    # instruction_strategy_1, 3 options -- no "hunter"/"other
    # claimant" category exists when there's only one claimant).
    strategy_state = build_public_state(day)
    if n_slots == 1:
        strategy = village_ai.choose_guard_instruction_strategy_1(strategy_state)
        kind = "guard_instruction_strategy_1"
    elif n_slots == 2:
        strategy = village_ai.choose_guard_instruction_strategy(strategy_state)
        kind = "guard_instruction_strategy"
    else:
        strategy = village_ai.choose_guard_instruction_strategy_3(strategy_state)
        kind = "guard_instruction_strategy_3"
    # A same-day re-run replaces the earlier assignment, so the earlier
    # (superseded) decision must not also be trained on.
    history[:] = [
        h for h in history
        if not (h["side"] == "village" and h["day"] == day
                and h["type"].startswith("guard_instruction_strategy"))
    ]
    log_decision("village", kind, strategy_state, strategy, day=day)

    # 2026-10-06 user request (information-asymmetry fix): the "medium"
    # and "seer" slots used to resolve to the TRUE Ps / Sy holder
    # (ROLES lookup), which a human villager can never know when a
    # rival claimant exists -- it handed the wolves' most likely bite
    # target to the verification step for free (and leaked the real
    # role's life/death through the gray fallback). They now resolve
    # from the PUBLIC CO status only: a random living claimant of that
    # role who isn't one of the watched Hn claimants. "gray" means
    # nobody who has claimed anything.
    def _pick_claimant(role_code):
        pool = [
            i for i in range(len(ROLES))
            if board[i] != "--" and co_history[i] == role_code and i not in watched
        ]
        return random.choice(pool) if pool else None

    medium_idx = _pick_claimant("Ps")
    medium_claimed_and_alive = medium_idx is not None
    seer_idx = _pick_claimant("Sy")
    seer_claimed_and_alive = seer_idx is not None

    slot_types = strategy.split("_")  # e.g. "medium_gray_hunter" -> [...]
    gray_candidates = [c for c in candidates if co_history[c] is None]
    targets = [None] * n_slots

    for slot in range(n_slots):
        kind = slot_types[slot]
        if kind == "medium" and medium_claimed_and_alive and medium_idx not in targets:
            targets[slot] = medium_idx
        elif kind == "seer" and seer_claimed_and_alive and seer_idx not in targets:
            targets[slot] = seer_idx
        elif kind == "hunter":
            # One of the OTHER watched claimants (never themselves).
            # This cyclic rotation reduces to the original "the other
            # one" (watched[1 - slot]) exactly when n_slots == 2.
            targets[slot] = watched[(slot + 1) % n_slots]
        # else falls through to the "gray" default below

    # Fill in anything still unset (either genuinely "gray", or a
    # fallback for an impossible/unavailable pick like "medium"
    # requested twice, or Medium not actually available).
    used = [t for t in targets if t is not None]
    for slot in range(n_slots):
        if targets[slot] is None:
            pool = [c for c in gray_candidates if c not in used]
            if not pool:
                pool = [c for c in candidates if c not in used]
            pick = random.choice(pool)
            targets[slot] = pick
            used.append(pick)

    guard_instructions = {watched[s]: targets[s] for s in range(n_slots)}

    if verbose:
        parts = [
            f"{PLAYERS[watched[s]]}(狩CO) → {PLAYERS[targets[s]]} を護衛"
            for s in range(n_slots)
        ]
        print("【護衛指示】" + " / ".join(parts))


def _shuffle_new_entries(lst, start):
    """Shuffle, in place, the entries appended since index `start` --
    the real claimant is always appended first, so list order must not
    be readable as a tell (2026-10-06, re-audit A7)."""
    tail = lst[start:]
    random.shuffle(tail)
    lst[start:] = tail


def _run_reports_shuffled(fns, day, verbose=True):
    """Run the report functions in their (logic-required) order, but
    print the resulting report lines in RANDOM order. 2026-10-06 (L7):
    the real claimant's report was always printed first, so the log
    (and anything later parsed from it) showed who is genuine."""
    if not verbose:
        _n_seer, _n_claims = len(seer_reports), len(medium_claims)
        for fn in fns:
            fn(day, verbose=False)
        _shuffle_new_entries(seer_reports, _n_seer)
        _shuffle_new_entries(medium_claims, _n_claims)
        return
    import io, contextlib
    _n_seer, _n_claims = len(seer_reports), len(medium_claims)
    lines = []
    for fn in fns:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(day, verbose=True)
        for ln in buf.getvalue().split("\n"):
            if not ln:
                continue
            if ln.startswith("   →") and lines:
                lines[-1] += "\n" + ln  # continuation stays with its report
            else:
                lines.append(ln)
    random.shuffle(lines)
    for ln in lines:
        print(ln)
    _shuffle_new_entries(seer_reports, _n_seer)
    _shuffle_new_entries(medium_claims, _n_claims)


def day_phase(day, verbose=True):

    if verbose:
        print()
        print(f"========== {day}日目 昼 ==========")

    # Report on YESTERDAY's execution / LAST NIGHT's investigation
    # before anyone acts today.
    # A dead nominator's old nomination no longer stands.
    for _nom in [n for n in nominations if board[n] == "--"]:
        del nominations[_nom]

    _begin_report_snapshot(day)
    _run_reports_shuffled([medium_report, fake_medium_report], day, verbose)
    _run_reports_shuffled([seer_report, fake_seer_report], day, verbose)

    # Anyone still silent decides today whether to reveal.
    medium_day_co(day, verbose=verbose)
    hunter_day_co(day, verbose=verbose)
    seer_day_co(day, verbose=verbose)

    # NOTE: full-game villager bluffing used to happen here
    # (villager_day_co) -- 500+ epochs showed it never settling into
    # a clear, useful pattern (win rate barely moved either way while
    # it kept oscillating between bluffs), so the call was removed
    # from day_phase, and the dead function itself was removed from
    # main.py entirely in a later cleanup pass. An earlier delivered
    # zip has both back if this is ever worth revisiting.

    # Wolves react next, after seeing whatever was just revealed
    # today. The Madman reacts LAST of all (madman_day_co) -- it has
    # no teammate to coordinate a claim with, so there's nothing lost
    # by letting it see the wolves' own same-day reaction too before
    # deciding whether/what to bluff.
    wolf_day1_joint_co(day, verbose=verbose)
    wolf_day_co(day, verbose=verbose)
    madman_day_co(day, verbose=verbose)

    # Every Day-1 CO claim (medium_day_co/hunter_day_co/seer_day_co's
    # real ones, plus wolf_day1_joint_co/wolf_day_co/madman_day_co's
    # fake ones) has now been buffered by announce_co(..., day=1) --
    # print them all together, grouped by role, plus the board-shape
    # notation. No-op (prints nothing) on any day other than 1, since
    # the buffer is only ever populated on Day 1.
    if day == 1:
        flush_day1_co_announcements(verbose=verbose)

    # Anyone (real Seer, fake wolf claimant, or fake Madman claimant)
    # who just revealed Sy FOR THE FIRST TIME today gets to report
    # immediately, in the same breath as their CO -- instead of
    # waiting for tomorrow's day_phase, which might never come if
    # they're killed tonight (see seer_report's NOTE for why that gap
    # mattered: it could silently drop even the day-0 free "random
    # white" result). Already-revealed claimants were already handled
    # by the earlier calls above, so this is a no-op for them.
    _run_reports_shuffled([seer_report, fake_seer_report], day, verbose)
    _end_report_snapshot()

    # STEPS 4/5 of the Madman addition (config.py's "Kg" comment):
    # once only 3 or 4 players remain, a genuine Villager gets first
    # crack at the endgame "村騙り" decoy bluff, then the Madman
    # decides whether to genuinely reveal -- seeing whether a decoy
    # already muddied the water today before committing (the same
    # "village-side act first, wolf-aligned side reacts" ordering
    # medium/hunter/seer -> wolf/madman already uses above). Both
    # write into the SAME self_declared_non_village list, which is
    # what makes a reveal seen there genuinely ambiguous -- see
    # madman_ai.choose_endgame_reveal_action / villager_ai.choose_
    # endgame_bluff_action for the full mechanic. Placed before
    # make_all_nominations/today's vote so either one can actually
    # influence today's outcome. wolf_endgame_declare goes last, after
    # the village-side decoy and the Madman's own reveal, so a wolf's
    # decision can react to whatever today's declarations already look
    # like (see wolf_endgame_declare's docstring for why a wolf gets
    # this option at all -- closing the "sole non-declarer is the
    # wolf" elimination hole a real played-out game exposed).
    villager_endgame_bluff(day, verbose=verbose)
    madman_endgame_reveal(day, verbose=verbose)
    wolf_endgame_declare(day, verbose=verbose)

    # NOW that every side's CO for today is settled, any Medium
    # claimant (real or fake) who hasn't nominated yet does so --
    # informed by whether a competing claim showed up today too.
    make_all_nominations(day, verbose=verbose)

    # NOTE: no board recap here -- the chronological CO announcements
    # above already show everything that happened today, and nobody's
    # alive/dead status changes during the day phase itself (only at
    # execution, right below, and at night). See show_board() calls
    # after execution and after the night's attack instead.

    # If 2+ Hunter claimants are now alive, set up tonight's
    # verification (see assign_guard_instructions' docstring).
    assign_guard_instructions(day, verbose=verbose)

    target = progression_ai(day, verbose=verbose)

    if verbose:
        print(f"処刑先: {PLAYERS[target]} ({ROLE_JP[ROLES[target]]})")
        print(f"村側の最善進行>>{_describe_execution_choice(target)}")

    execute_player(target)

    global last_execution_target
    last_execution_target = target


# ============================================================
# Werewolf Attack
# ============================================================

# ============================================================
# Hunter Guard
# ============================================================

def hunter_guard(day):

    global guard_target, last_guard_choice

    hunter_alive = any(
        role == "Hn" and board[i] != "--"
        for i, role in enumerate(ROLES)
    )

    if not hunter_alive:
        guard_target = None
        return None

    true_hunter_idx = next(
        i for i, role in enumerate(ROLES)
        if role == "Hn" and board[i] != "--"
    )

    # Under guard-instruction verification, the real Hunter (if one
    # of the currently-watched claimants) must guard their ASSIGNED
    # target instead of freely choosing -- that's what makes the
    # verification meaningful. (If that target already died from
    # today's execution before night falls, there's nothing left to
    # guard -- fall through to a normal free choice instead.)
    if true_hunter_idx in guard_instructions and board[guard_instructions[true_hunter_idx]] != "--":
        target = guard_instructions[true_hunter_idx]
        # forced by the verification instruction -- not the policy's own
        # choice, so it is NOT logged/trained (re-audit A3).
        guard_target = target
        last_guard_choice = target
        return target

    state = build_hunter_state(day)

    target = hunter_ai.choose_guard_target(state)
    log_decision("hunter", "guard", state, target, day=day)

    guard_target = target
    last_guard_choice = target
    return target


def seer_night_investigate(day, verbose=True):
    """
    If the real Seer is alive, they privately investigate exactly one
    OTHER living player tonight and learn that player's true
    alignment immediately -- regardless of whether they've revealed
    yet (revealing only controls whether/when this becomes PUBLIC via
    seer_report; the private ability itself runs every night the Seer
    is alive, matching real Werewolf play).
    """

    global seer_investigations

    seer_index = next((i for i, role in enumerate(ROLES) if role == "Sy"), None)
    if seer_index is None or board[seer_index] == "--":
        return

    candidates = [i for i, s in enumerate(board) if s != "--" and i != seer_index]
    if not candidates:
        return

    state = build_seer_state(day, seer_index, candidates)
    target = seer_ai.choose_investigate_target(state)
    log_decision("seer", "investigate", state, target, day=day)

    is_wolf = ROLES[target] == "Wo"
    seer_investigations.append({"day": day, "target": target, "is_wolf": is_wolf})

    if verbose:
        # NOTE: no "(結果は非公開)" qualifier here -- night-phase actions
        # are inherently private to begin with (nobody but the Seer
        # themself is watching this happen), so there's nothing to
        # contrast against. Whether/when the result BECOMES public is
        # entirely seer_report's/seer_day_co's call, made visible at
        # that point instead (an explicit "reveal" or "silent" choice
        # each day, not decided here).
        print(f"{PLAYERS[seer_index]}(占い)が{PLAYERS[target]}を占った")


# ============================================================
# Werewolf Attack
# ============================================================

def werewolf_attack(day):

    living_wolves = [
        i for i, role in enumerate(ROLES)
        if role == "Wo" and board[i] != "--"
    ]

    if not living_wolves:
        return None

    targets = [
        i for i, role in enumerate(ROLES)
        if role != "Wo" and board[i] != "--"
    ]

    state = build_wolf_state(day=day, targets=targets)

    target = wolf_ai.choose_attack_target(state)
    log_decision("wolf", "attack", state, target, day=day)

    return target


def resolve_night_kill(target, verbose=True):
    """
    Apply the wolves' chosen attack, unless the hunter guarded
    that exact player tonight.

    Returns (actual_kill, announce_confirmed_white) where the
    second value is the target to announce as confirmed white
    RIGHT NOW (if the Hunter had already revealed before this
    guard success happened), or None otherwise -- left to the
    caller to print at the right point in the log.
    """

    global last_guard_saved, guard_ever_succeeded
    global guard_success_count, guard_success_targets, attack_victims, last_night_victim

    if target is None:
        last_guard_saved = False
        return None, None

    if guard_target is not None and target == guard_target:
        last_guard_saved = True
        guard_ever_succeeded = True
        guard_success_count += 1
        guard_success_targets.append(guard_target)

        real_hunter_index = next(
            (i for i, role in enumerate(ROLES) if role == "Hn"), None
        )
        already_revealed = (
            real_hunter_index is not None and co_history[real_hunter_index] == "Hn"
        )

        # BUG FIX: this used to announce "confirmed white" purely off
        # already_revealed -- i.e. the moment the real Hunter had
        # CO'd, with NO check for whether their claim is actually
        # TRUSTED yet. That's inconsistent with build_public_state's
        # own confirmed_villagers logic just below (and everywhere
        # else in the file): with 2+ Hunter claimants, a guard-success
        # declaration is only trustworthy once guard-instruction
        # verification has confirmed WHICH claimant is genuine
        # (confirmed_real_hunter) -- an unverified contested claimant
        # could still be a lying wolf. Printing "確定で白!" regardless
        # told the reader (and, via last_execution_target/history, the
        # log) something stronger than the AI itself actually believed
        # -- the village_ai side correctly left that target OUT of
        # confirmed_villagers in this exact situation, so it kept
        # voting to execute someone the log had just called "confirmed
        # white". The condition below mirrors build_public_state's
        # num_hunter_claimants<=1 / confirmed_real_hunter check exactly,
        # so what gets announced here always matches what the rest of
        # the game actually treats as confirmed.
        # 2026-10-06 (L8): mirrors build_public_state's current rule --
        # a lone claimant is NOT trusted; only a publicly proven Hunter.
        hunter_trusted = already_revealed and confirmed_real_hunter == real_hunter_index
        announce = guard_target if hunter_trusted else None

        return None, announce  # attack failed -- no one dies

    last_guard_saved = False
    board[target] = "--"
    attack_victims.append(target)
    last_night_victim = target
    return target, None


# ============================================================
# Night Phase
# ============================================================

def resolve_guard_instructions(attack_target, verbose=True):
    """
    After the night resolves: if the wolves' attack landed on one
    of the two assigned watch-targets, we learn something for free.
    Only the REAL Hunter's guard actually blocks an attack, so:
      - if the assigned target DIED, that claimant's "guard" did
        nothing -- they are proven fake.
      - if the assigned target SURVIVED the attack, a real guard
        blocked it -- that claimant is proven genuine.
    """

    global confirmed_real_hunter

    if not guard_instructions or attack_target is None:
        return

    for claimant, assigned_target in guard_instructions.items():

        if assigned_target != attack_target:
            continue

        if board[claimant] == "--":
            # The claimant THEMSELVES died (e.g. executed the same
            # day these instructions were assigned) before tonight
            # even happened. A dead claimant obviously can't have
            # protected anyone -- real or fake -- so their assigned
            # target dying tells us NOTHING about their identity.
            # Skip verification entirely rather than wrongly
            # concluding "fake" just because nobody was watching.
            continue

        if board[assigned_target] != "--":
            # 2026-10-06 (round 4): the assigned target SURVIVED. Whether
            # it was attacked at all is hidden from the public (a
            # peaceful night looks the same whichever target the wolves
            # chose, and a free guard by a silent real Hunter can also
            # produce it), so nothing may be concluded about this
            # claimant. Only a visible DEATH of the assigned target
            # (public) can expose a fake. A real Hunter is only ever
            # proven through public counting (build_public_state).
            continue
        else:
            if claimant not in confirmed_fake_hunters:
                confirmed_fake_hunters.append(claimant)
                if verbose:
                    print(f"{PLAYERS[claimant]} の護衛先が死亡 → "
                          f"{PLAYERS[claimant]} は狩人偽物と判明!")


def night_phase(day, verbose=True):

    if verbose:
        print()
        print(f"========== {day}日目 夜 ==========")

    # The hunter chooses privately, without seeing the wolves'
    # target (and vice versa) -- order here doesn't leak anything
    # since neither side's build_*_state reads the other's choice.
    global last_night_victim
    last_night_victim = None
    hunter_guard(day)

    if verbose and guard_target is not None:
        print(f"狩人の護衛先: {PLAYERS[guard_target]}")

    # Independent of the hunter/wolf actions above -- neither of their
    # build_*_state functions reads anything the Seer's private
    # investigation touches, so call order here doesn't leak anything
    # either way.
    seer_night_investigate(day, verbose=verbose)

    attack_target = werewolf_attack(day)
    actual_kill, announce_white = resolve_night_kill(attack_target, verbose=verbose)
    if last_guard_saved:
        _extend_fake_declarations(day, verbose=False)
    resolve_guard_instructions(attack_target, verbose=verbose)

    if verbose:
        if attack_target is not None:
            print(f"人狼の襲撃先: {PLAYERS[attack_target]}")
            print(f"狼側の最善進行>>{_describe_attack_choice(attack_target)}")
            if actual_kill is None:
                print("→ 護衛成功!襲撃は失敗した")
                if announce_white is not None:
                    real_hunter_index = next(
                        (i for i, role in enumerate(ROLES) if role == "Hn"), None
                    )
                    print(f"   → {PLAYERS[real_hunter_index]} が護衛先を公表: "
                          f"{PLAYERS[announce_white]} は確定で白!")
            else:
                print(f"→ {PLAYERS[attack_target]} が襲撃された")
        print()
        show_board()

    return check_game_end()


# ============================================================
# Main Game
# ============================================================

def run_game(verbose=True, show_role_candidates=False):
    """Wrapper: in verbose mode the printed 村側の最善進行 lines are
    corrected after the game, once it is known whether each ロラ was
    carried through (see _describe_execution_choice)."""
    if not verbose:
        return _run_game_inner(verbose, show_role_candidates)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = _run_game_inner(verbose, show_role_candidates)
    # 2026-10-05 user request: a "◯◯ロラ" on the LAST execution of the
    # game was never carried through (the game ended first), so it is
    # really a 決め打ち too.
    if _label_history and _label_history[-1]["is_roll"]:
        _label_history[-1]["label"] = _label_history[-1]["label"][:-2] + "決め打ち"
        _label_history[-1]["is_roll"] = False
    lines = buf.getvalue().split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("村側の最善進行>>")]
    if len(idx) == len(_label_history):
        for i, e in zip(idx, _label_history):
            lines[i] = "村側の最善進行>>" + e["label"]
    print("\n".join(lines), end="")
    return result


def _run_game_inner(verbose=True, show_role_candidates=False):

    reset_game()

    if verbose:
        print("=" * 50)
        print("　　　人狼ゲーム開始")
        print("=" * 50)
        show_roles()

    day = 1
    winner = None

    while True:

        # Day
        day_phase(day, verbose=verbose)

        # NOTE: the original script only checked for game-end after
        # the night phase. That meant a day-execution which already
        # decided the winner (e.g. the last wolf executed) was missed
        # until an extra, meaningless night played out. Checking here
        # fixes that.
        winner = check_game_end()
        if winner is not None:
            break

        # Night
        winner = night_phase(day, verbose=verbose)
        if winner is not None:
            break

        day += 1

    if verbose:
        winner_jp = "人狼陣営" if winner == "Werewolf" else "村人陣営"
        print(f"勝者: {winner_jp}")

        if show_role_candidates:
            candidates = get_role_candidates()
            print()
            print("【役職候補(村視点での推測範囲)】")
            for i, candidate in enumerate(candidates):
                jp_candidates = [ROLE_JP.get(c, c) for c in candidate]
                print(f"  {PLAYERS[i]} : " + " / ".join(jp_candidates))

        print()
        print("=" * 50)
        print("　　　人狼ゲーム終了")
        print("=" * 50)

    return {
        "winner": winner,     # "Werewolf" or "Village"
        "history": history,   # list of {side, type, day, state, action}
        "roles": list(ROLES),
        "execution_history": list(execution_history),  # player index executed each day, in order
        # Every Day-1 CO claim this game, as (player_idx, claimed_role_
        # code) pairs -- same co_history/claim_day==1 computation as
        # _current_day1_config(), just kept at the per-claimant level
        # instead of collapsed into a (sy,ps,hn) count. Paired with
        # "roles" above, this is what lets a caller reconstruct the
        # FULL real/Werewolf/Madman breakdown of this one game's board
        # after the fact (board_split.format_day1_config_detailed),
        # not just its claim counts -- added 2026-10-02 so train_and_
        # report.py's epoch-progress display can show the actual top
        # compositions behind a count bucket like (2,2,2), not just
        # the count itself.
        "day1_claims": [
            (i, co) for i, co in enumerate(co_history)
            if co is not None and claim_day[i] == 1
        ],
    }


def run_forced_day1_game(forced_claims, verbose=True, show_role_candidates=False):
    """
    Like run_game(), except Day 1's claims are DICTATED instead of
    decided by hunter_ai.choose_co_action / wolf_ai.day1_joint_co_
    policy / madman_ai.co_policy -- every decision from Day 1 voting
    onward (nominations, guard instructions, execution voting, night
    attacks, endgame declarations, everything) still goes entirely
    through the real, currently-loaded policies, completely untouched.

    Why this exists (2026-10-02): once a Day-1 board configuration
    stops being something the (shared, unsplit) claim-generating
    policies ever produce on their own, find_matching_games() can
    search millions of self-played games and never find one -- the
    per-board-config DECISION weights for that configuration can be
    sitting there fully intact (board_split's selective decay means
    an untouched bucket never forgets), but normal self-play simply
    never reaches it anymore to demonstrate that. This sidesteps the
    claim-generation step entirely for one test game, so a config's
    frozen bucket can still be inspected/played out even after the
    rest of training has permanently moved past it.

    forced_claims : list of (player_idx, role_code) pairs, role_code
    one of "Sy"/"Ps"/"Hn". The real holder of a role does not have to
    be included -- a role nobody claims this way is simply left
    silent, exactly like any other game. Typical use: look up the
    real holders via ROLES.index(...)/[i for i,r in enumerate(ROLES)
    if r == "Wo"] after reset_game() has picked this game's roles,
    then hand-assemble the exact claim pattern you want to inspect
    (see board_split.format_day1_config_detailed's labels for how a
    pattern like "真狼-真狂-真狼" maps onto which claimant is real/
    Wo/Kg per role slot).

    NOT suitable for training -- this is a one-off inspection tool.
    update_policies()/decay_all_policies() still only ever see what
    ACTUALLY happened during ordinary self-play, so calling this does
    not feed anything back into the weights; it only lets you watch
    what the currently-loaded policies already do with the board
    shape you hand them.
    """
    reset_game()

    if verbose:
        print("=" * 50)
        print("　　　人狼ゲーム開始(1日目の配役を強制)")
        print("=" * 50)
        show_roles()
        print()
        print("========== 1日目 昼 ==========")

    for i, role in forced_claims:
        board[i] = role
        co_history[i] = role
        claim_day[i] = 1
        announce_co(i, role, verbose=verbose, day=1)

    flush_day1_co_announcements(verbose=verbose)
    _begin_report_snapshot(1)
    seer_report(1, verbose=verbose)
    fake_seer_report(1, verbose=verbose)
    _end_report_snapshot()
    villager_endgame_bluff(1, verbose=verbose)
    madman_endgame_reveal(1, verbose=verbose)
    wolf_endgame_declare(1, verbose=verbose)
    make_all_nominations(1, verbose=verbose)
    assign_guard_instructions(1, verbose=verbose)

    target = progression_ai(1, verbose=verbose)
    if verbose:
        print(f"処刑先: {PLAYERS[target]} ({ROLE_JP[ROLES[target]]})")
        print(f"村側の最善進行>>{_describe_execution_choice(target)}")
    execute_player(target)

    global last_execution_target
    last_execution_target = target

    winner = check_game_end()
    day = 1
    while winner is None:
        winner = night_phase(day, verbose=verbose)
        if winner is not None:
            break
        day += 1
        day_phase(day, verbose=verbose)
        winner = check_game_end()

    if verbose:
        winner_jp = "人狼陣営" if winner == "Werewolf" else "村人陣営"
        print(f"勝者: {winner_jp}")

        if show_role_candidates:
            candidates = get_role_candidates()
            print()
            print("【役職候補(村視点での推測範囲)】")
            for i, candidate in enumerate(candidates):
                jp_candidates = [ROLE_JP.get(c, c) for c in candidate]
                print(f"  {PLAYERS[i]} : " + " / ".join(jp_candidates))

        print()
        print("=" * 50)
        print("　　　人狼ゲーム終了")
        print("=" * 50)

    return {
        "winner": winner,
        "history": history,
        "roles": list(ROLES),
        "execution_history": list(execution_history),
        "day1_claims": [
            (i, co) for i, co in enumerate(co_history)
            if co is not None and claim_day[i] == 1
        ],
    }


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    # Load whatever train_and_report.py has saved so far, so a plain
    # `python main.py` demo game actually reflects the trained
    # policies -- without this, every run used all-zero (uniform
    # random) weights regardless of how much training had been done,
    # since nothing else in this file ever touched persistence.py.
    import persistence
    persistence.load_weights()
    run_game()

"""小さな配役における CO 戦略の厳密計算（研究ノート「小さな配役におけるCO戦略の厳密解析」）。

村側は「進行」（霊媒師・狩人の CO 日、護衛方針、対抗 CO、ローラー）を事前に決める。人狼はそれを知っている。
人狼は互いが人狼であることだけを知り、誰が霊媒師・狩人かは知らない。CO・霊能結果・平和な朝
（自分たちの襲撃が護衛された）から、ベイズの定理で「誰が霊媒師・狩人か」の確率を更新し、
その情報だけに基づいて、村の勝率を最も下げる行動（騙り CO、偽の霊能結果、襲撃先）を選ぶ。
これは、進行を固定したときの人狼の最善応答（部分観測マルコフ決定過程の最適方策）であり、
乱数を使わず有理数で厳密に計算する。

ルール（昼開始）
  朝   : 霊媒師 CO 者が前日の処刑者の結果を発表する（騙りの人狼は白黒を自由に選ぶ）。
  CO   : (a) 予定日に達した本物の霊媒師・狩人が CO する（CO の日と順番は公開される）
         (b) 人狼が騙り CO するかを決める（(a) を見てから）
         (c) 自分の役職を騙られた本物が、その場で対抗 CO する（「対抗あり」の進行のみ）
         遅れて CO した霊媒師（騙りを含む）は、それまでの処刑者全員の結果を発表する。
  処刑 : 村は公開情報だけから処刑先を決める（下の「段」の順に選び、同じ段の中では一様ランダム）。
  夜   : 狩人が自分以外の 1 人を護衛し、人狼が人狼以外の 1 人を襲撃する。護衛先を襲撃すると誰も死なない。
  勝敗 : 人狼 0 人で村の勝ち。人狼の数 ≥ 人狼以外の数 で人狼の勝ち（処刑後・襲撃後に判定）。

村の推理は「論理的に確定すること」だけを使う。公開情報（CO、霊能結果、襲撃死、平和な朝、
ゲームが続いていること）と村側の進行に矛盾しない役職の割り当てをすべて列挙し、
すべての割り当てで人狼なら「確定黒」、どの割り当てでも人狼でなければ「確定白」とする。
処刑の段: 確定黒 → （ローラーありなら）対抗のある役職の CO 者 → 灰（CO していない人）
          → （ローラーなしなら）対抗のある役職の CO 者 → 残りの CO 者 → 確定白以外 → 全員

Python 3.8 以降、標準ライブラリのみ。
"""
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
from typing import NamedTuple, Optional

NEVER = None


class Progression(NamedTuple):
    """村側の進行。"""
    d_m: Optional[int] = 1        # 霊媒師が CO する日（None = 自分からは CO しない）
    d_h: Optional[int] = 1        # 狩人が CO する日（None = 自分からは CO しない）
    guard: str = "random"         # "none"（護衛なし）/ "random"（自分以外から一様）/ "medium"（霊媒師 CO 者を優先）
    roller: bool = True           # 対抗のある役職の CO 者を、灰より先に処刑するか
    counter: bool = True          # 自分の役職を騙られたら、その日に対抗 CO するか

    def label(self):
        f = lambda d: "なし" if d is None else f"{d}日目"
        g = {"none": "護衛なし", "random": "ランダム護衛", "medium": "霊媒師護衛"}[self.guard]
        parts = []
        if self.d_m is not None or self.guard == "medium":
            parts.append(f"霊CO {f(self.d_m)}")
        parts += [f"狩CO {f(self.d_h)}", g]
        if self.roller:
            parts.append("ローラー")
        if not self.counter:
            parts.append("対抗なし")
        return "・".join(parts)


class Cast(NamedTuple):
    name: str
    roles: tuple                  # 席ごとの役職 "W"（人狼）/ "M"（霊媒師）/ "H"（狩人）/ "V"（村人）


CAST_7 = Cast("霊狩2w", ("W", "W", "M", "H", "V", "V", "V"))
CAST_4 = Cast("村村狩狼", ("W", "H", "V", "V"))


def solve(cast, prog, wolf_mode="best", exact=True, force_claims=None):
    """村の勝率を返す（exact=True なら Fraction、False なら float で高速に計算）。

    wolf_mode:
      "best"        人狼は騙り CO・偽の霊能結果・襲撃先を、自分たちの情報の範囲で最善に選ぶ
      "silent"      騙り CO はしない。襲撃先は最善に選ぶ
      "log"         対話ログのシミュレータの人狼: 騙りなし、襲撃は CO 済みの狩人 > CO 済みの霊媒師 > 一様
      "log_giveup"  上と同じだが、霊媒師への襲撃が護衛で 1 回失敗したら、以後は人狼以外から一様に襲撃する
    force_claims: {日: 騙る役職のタプル} を与えると、その日の人狼の騙り CO をそれに固定する（分岐ごとの値を見るため）。
                  例: {1: ("H",)} = 1 日目に人狼 1 人が狩人を騙る。その他の行動は wolf_mode に従う。
    """
    F = Fraction if exact else (lambda a, b=1: a / b)
    roles = cast.roles
    n = len(roles)
    seats = tuple(range(n))
    W = frozenset(i for i in seats if roles[i] == "W")
    nonwolf = [i for i in seats if i not in W]
    has_m = "M" in roles
    n_w = len(W)

    # 人狼から見た「世界」= (霊媒師の席, 狩人の席)。最初は一様。
    worlds0 = []
    for h in nonwolf:
        for m in (nonwolf if has_m else [None]):
            if m != h:
                worlds0.append((m, h))
    p0 = F(1, len(worlds0))
    WORLDS0 = tuple((w, p0) for w in sorted(worlds0, key=str))

    # 村の論理判定用: 人狼の集合の全候補
    wolf_sets = [frozenset(c) for c in combinations(seats, n_w)]

    def winner(alive):
        w = len(alive & W)
        if w == 0:
            return F(1)
        if w >= len(alive) - w:
            return F(0)
        return None

    # ------------------------------------------------------------ 村の論理
    def protocol_claim_day(role, seat, claims, alive_at, last_day):
        d_sched = prog.d_m if role == "M" else prog.d_h
        others = [c[1] for s, c in enumerate(claims) if c is not None and c[0] == role and s != seat]
        for d in range(1, last_day + 1):
            if seat not in alive_at[d]:
                return None
            if d_sched is not None and d == d_sched:
                return d
            if prog.counter and any(od <= d for od in others):
                return d
        return None

    @lru_cache(maxsize=None)
    def labels(day, alive, claims, reports, executed, nights):
        """公開情報から論理的に分かる (確定黒, 確定白)。

        (人狼の集合, 霊媒師, 狩人) の割り当てのうち、次をすべて満たすものを「矛盾しない」とする。
          - 襲撃で死んだ人は人狼ではない。平和な朝の前夜には、本物の狩人が生きていた。
          - 各時点でゲームが続いていた（人狼 1 人以上、人狼 < 人狼以外）。
          - 人狼以外の CO はすべて本物で、CO した日は村側の進行どおり。
            CO していない本物の霊媒師・狩人は、進行上まだ CO しないはずの人。
          - 本物の霊媒師の結果は正しい。
        """
        alive_at = {}
        events = []                         # (夜の死者 or None or "exec", その後の生存者)
        peaceful_before = []                # 平和な朝の前夜の生存者（狩人はこの中にいた）
        cur = frozenset(seats)
        for d in range(1, day + 1):
            alive_at[d] = cur
            if d - 1 < len(executed):
                cur = cur - {executed[d - 1]}
                events.append(("exec", cur))
            if d - 1 < len(nights):
                v = nights[d - 1]
                if v is None:
                    peaceful_before.append(cur)
                else:
                    cur = cur - {v}
                events.append((v, cur))
        victims = frozenset(v for v, _ in events if v not in ("exec", None))
        # 本物ならいつ CO するはずか（人狼の割り当てによらない）
        expect = {(r, s): protocol_claim_day(r, s, claims, alive_at, day) for r in ("M", "H") for s in seats}
        must_h = frozenset(seats)
        for b in peaceful_before:
            must_h &= b
        consistent = []
        for wolves in wolf_sets:
            if wolves & victims:
                continue
            ok = True
            for _, after in events:
                w = len(after & wolves)
                if w == 0 or w >= len(after) - w:
                    ok = False
                    break
            if not ok:
                continue
            cand = {}
            for r in (("M", "H") if has_m else ("H",)):
                claimants = [s for s in seats if s not in wolves and claims[s] is not None and claims[s][0] == r]
                if len(claimants) >= 2:
                    ok = False
                    break
                if claimants:
                    s0 = claimants[0]
                    _, d0, q0 = claims[s0]
                    sched = prog.d_m if r == "M" else prog.d_h
                    # 予定日以外の CO は対抗 CO なので、同じ役職の CO が先にあったはず
                    by_counter = any(claims[t] is not None and claims[t][0] == r and claims[t][2] < q0
                                     for t in seats if t != s0)
                    ok_order = d0 == sched or by_counter
                    cand[r] = [s0] if d0 == expect[(r, s0)] and ok_order else []
                else:
                    cand[r] = [s for s in seats if s not in wolves and claims[s] is None and expect[(r, s)] is None]
            if not ok:
                continue
            hs = [h for h in cand["H"] if h in must_h]
            if has_m:
                ms = []
                for m in cand["M"]:
                    if all((col == "B") == (x in wolves) for (x, col) in reports[m]):
                        ms.append(m)
                if not any(m != h for m in ms for h in hs):
                    continue
            elif not hs:
                continue
            consistent.append(wolves)
        assert consistent, "公開情報と矛盾しない割り当てがない"
        black = frozenset(s for s in alive if all(s in w for w in consistent))
        white = frozenset(s for s in alive if all(s not in w for w in consistent))
        return black, white

    def exec_candidates(day, alive, claims, reports, executed, nights):
        black, white = labels(day, alive, claims, reports, executed, nights)
        by_role = defaultdict(list)
        for s in seats:
            if claims[s] is not None:
                by_role[claims[s][0]].append(s)
        contested = frozenset(s for ss in by_role.values() if len(ss) >= 2 for s in ss)
        claimed = frozenset(s for s in seats if claims[s] is not None)
        cont = (contested & alive) - white
        gray = (alive - claimed) - white
        rest = (alive & claimed) - white
        tiers = [black] + ([cont, gray] if prog.roller else [gray, cont]) + [rest, alive - white, alive]
        for t in tiers:
            if t:
                return sorted(t)

    # ------------------------------------------------------------ 補助
    def set_claim(claims, s, role, day):
        """CO を記録する。(役職, 日, 何番目の CO か) — CO の順番は公開情報。"""
        c = list(claims)
        c[s] = (role, day, sum(1 for x in claims if x is not None))
        return tuple(c)

    def add_reports(reports, s, items):
        r = list(reports)
        r[s] = r[s] + tuple(items)
        return tuple(r)

    def normalize(items):
        """[(world, weight)] → (正規化した worlds タプル, 合計の重み)。"""
        acc = defaultdict(lambda: F(0))
        for w, p in items:
            if p:
                acc[w] += p
        tot = sum(acc.values())
        return tuple(sorted(((w, p / tot) for w, p in acc.items()), key=lambda t: str(t[0]))), tot

    def split(worlds, f):
        """各世界を f(w)（公開される結果）で分け、{結果: (worlds, 確率)} を返す。"""
        groups = defaultdict(list)
        for w, p in worlds:
            groups[f(w)].append((w, p))
        return {k: normalize(v) for k, v in groups.items()}

    def wmin(values):
        best = None
        for v in values:
            if best is None or v < best:
                best = v
        return best

    # ------------------------------------------------------------ 朝
    @lru_cache(maxsize=None)
    def morning(day, alive, claims, reports, executed, nights, worlds, gave_up):
        if day == 1:
            return co_a(day, alive, claims, reports, executed, nights, worlds, gave_up)
        x = executed[-1]
        col = "B" if x in W else "W"

        def real_report(w):
            m = w[0]
            if m is not None and m in alive and claims[m] is not None and claims[m][0] == "M":
                return add_reports(reports, m, [(x, col)])
            return reports
        total = F(0)
        for rep, (ws, p) in split(worlds, real_report).items():
            total += p * fake_reports(day, alive, claims, rep, executed, nights, ws, gave_up, x)
        return total

    def fake_reports(day, alive, claims, reports, executed, nights, worlds, gave_up, x):
        fake = [s for s in sorted(alive & W) if claims[s] is not None and claims[s][0] == "M"]
        if not fake:
            return co_a(day, alive, claims, reports, executed, nights, worlds, gave_up)
        opts = []
        for cols in product("BW", repeat=len(fake)):
            r = reports
            for s, c in zip(fake, cols):
                r = add_reports(r, s, [(x, c)])
            opts.append(co_a(day, alive, claims, r, executed, nights, worlds, gave_up))
        return wmin(opts)

    # ------------------------------------------------------------ CO
    def real_claims(day, alive, claims, reports, executed, w, pred):
        """世界 w で、条件 pred を満たす本物が CO したあとの (claims, reports)。"""
        m, h = w
        for role, seat in (("M", m), ("H", h)):
            if seat is not None and seat in alive and claims[seat] is None and pred(role, seat, claims):
                claims = set_claim(claims, seat, role, day)
                if role == "M":
                    reports = add_reports(reports, seat, [(x, "B" if x in W else "W") for x in executed])
        return claims, reports

    def co_a(day, alive, claims, reports, executed, nights, worlds, gave_up):
        sched = lambda role, seat, c: (prog.d_m if role == "M" else prog.d_h) == day
        total = F(0)
        groups = split(worlds, lambda w: real_claims(day, alive, claims, reports, executed, w, sched))
        for (c, r), (ws, p) in groups.items():
            total += p * co_b(day, alive, c, r, executed, nights, ws, gave_up)
        return total

    @lru_cache(maxsize=None)
    def co_b(day, alive, claims, reports, executed, nights, worlds, gave_up):
        free = [s for s in sorted(alive & W) if claims[s] is None]
        forced = (force_claims or {}).get(day)
        if forced is not None:
            c = claims
            for s, role in zip(free, forced):
                if role is not None:
                    c = set_claim(c, s, role, day)
            return co_c(day, alive, c, reports, executed, nights, worlds, gave_up)
        if wolf_mode != "best" or not free:
            return co_c(day, alive, claims, reports, executed, nights, worlds, gave_up)
        opts = []
        # まだ CO していない生存人狼どうしは、公開情報の上で区別がつかない（対称）ので、
        # 「どの役職を何人が騙るか」の組み合わせだけを調べればよい。
        menu = [None, "M", "H"] if has_m else [None, "H"]
        for combo in combinations_with_replacement(menu, len(free)):
            c = claims
            fake_m = []
            for s, role in zip(free, combo):
                if role is not None:
                    c = set_claim(c, s, role, day)
                    if role == "M":
                        fake_m.append(s)
            k = len(fake_m) * len(executed)
            for cols in product("BW", repeat=k):
                r = reports
                i = 0
                for s in fake_m:
                    r = add_reports(r, s, [(x, cols[i + j]) for j, x in enumerate(executed)])
                    i += len(executed)
                opts.append(co_c(day, alive, c, r, executed, nights, worlds, gave_up))
        return wmin(opts)

    def co_c(day, alive, claims, reports, executed, nights, worlds, gave_up):
        if not prog.counter:
            return execution(day, alive, claims, reports, executed, nights, worlds, gave_up)

        def countered(role, seat, c):
            return any(c[s] is not None and c[s][0] == role for s in seats if s != seat)
        total = F(0)
        groups = split(worlds, lambda w: real_claims(day, alive, claims, reports, executed, w, countered))
        for (c, r), (ws, p) in groups.items():
            total += p * execution(day, alive, c, r, executed, nights, ws, gave_up)
        return total

    # ------------------------------------------------------------ 処刑と夜
    @lru_cache(maxsize=None)
    def execution(day, alive, claims, reports, executed, nights, worlds, gave_up):
        cands = exec_candidates(day, alive, claims, reports, executed, nights)
        total = F(0)
        for x in cands:
            a = alive - {x}
            r = winner(a)
            if r is None:
                # 処刑された人の役職は公開されないが、人狼は「その人が霊媒師・狩人だった世界」を
                # 以後の CO などから区別できる。ここでは世界の集合をそのまま引き継ぐ。
                r = night(day, a, claims, reports, executed + (x,), nights, worlds, gave_up)
            total += r
        return total / len(cands)

    def guard_dist(alive, claims, h):
        if h is None or h not in alive or prog.guard == "none":
            return {None: F(1)}
        others = sorted(alive - {h})
        if prog.guard == "medium":
            meds = [s for s in others if claims[s] is not None and claims[s][0] == "M"]
            if meds:
                others = meds
        return {s: F(1, len(others)) for s in others}

    def attack_value(day, alive, claims, reports, executed, nights, worlds, gave_up, t, giveup_rule=None):
        """襲撃先 t のときの村の勝率（世界と護衛について平均）。"""
        outcomes = defaultdict(list)       # 公開される結果（平和な朝かどうか）→ [(world, weight)]
        for w, p in worlds:
            for g, pg in guard_dist(alive, claims, w[1]).items():
                outcomes[g == t].append((w, p * pg))
        total = F(0)
        for peaceful, items in outcomes.items():
            ws, pr = normalize(items)
            gu = gave_up or (giveup_rule is not None and peaceful and giveup_rule(t))
            if peaceful:
                total += pr * morning(day + 1, alive, claims, reports, executed, nights + (None,), ws, gu)
            else:
                a = alive - {t}
                r = winner(a)
                if r is None:
                    r = morning(day + 1, a, claims, reports, executed, nights + (t,), ws, gu)
                total += pr * r
        return total

    def night(day, alive, claims, reports, executed, nights, worlds, gave_up):
        targets = sorted(alive - W)
        if wolf_mode in ("log", "log_giveup"):
            known_h = [s for s in targets if claims[s] is not None and claims[s][0] == "H"]
            known_m = [s for s in targets if claims[s] is not None and claims[s][0] == "M"]
            if wolf_mode == "log_giveup" and gave_up:
                known_m = []
            if known_h:
                choice = {known_h[0]: F(1)}
            elif known_m:
                choice = {known_m[0]: F(1)}
            else:
                choice = {s: F(1, len(targets)) for s in targets}
            rule = (lambda t: t in known_m) if wolf_mode == "log_giveup" else None
            return sum(pt * attack_value(day, alive, claims, reports, executed, nights, worlds, gave_up, t, rule)
                       for t, pt in choice.items())
        return wmin(attack_value(day, alive, claims, reports, executed, nights, worlds, gave_up, t)
                    for t in targets)

    empty = tuple(None for _ in seats)
    no_rep = tuple(() for _ in seats)
    return morning(1, frozenset(seats), empty, no_rep, (), (), WORLDS0, False)


def all_progressions(cast):
    """探索する進行の全体。"""
    days = [1, 2, 3, NEVER]
    if "M" in cast.roles:
        for d_m in days:
            for d_h in days:
                for guard in ("random", "medium"):
                    for roller in (True, False):
                        yield Progression(d_m, d_h, guard, roller, True)
    else:
        for d_h in days:
            for roller in (True, False):
                for counter in (True, False):
                    yield Progression(NEVER, d_h, "random", roller, counter)

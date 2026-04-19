import math
import heapq
import time
import sys
import random

EPS = 1e-9
TLIM = 45.0


def euclid(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def read_input(tokens):
    ptr = 0
    n = int(float(tokens[ptr]))
    ptr += 1
    m = int(float(tokens[ptr]))
    ptr += 1
    facs = []
    for _ in range(n):
        setup = float(tokens[ptr])
        ptr += 1
        cap = float(tokens[ptr])
        ptr += 1
        x = float(tokens[ptr])
        ptr += 1
        y = float(tokens[ptr])
        ptr += 1
        facs.append((setup, cap, x, y))
    custs = []
    for _ in range(m):
        demand = float(tokens[ptr])
        ptr += 1
        x = float(tokens[ptr])
        ptr += 1
        y = float(tokens[ptr])
        ptr += 1
        custs.append((demand, x, y))
    return facs, custs


def build_cands(facs, custs, k):
    cand = []
    for _, cx, cy in custs:
        arr = [(euclid(cx, cy, fx, fy), j) for j, (_, _, fx, fy) in enumerate(facs)]
        cand.append(heapq.nsmallest(k, arr))
    return cand


def build_inv(cand, n):
    inv = [[] for _ in range(n)]
    for i, cands in enumerate(cand):
        for d, j in cands:
            inv[j].append((i, d))
    return inv


def build_dist(facs, custs):
    return [[euclid(cx, cy, fx, fy) for _, _, fx, fy in facs] for _, cx, cy in custs]


def dist(D, custs, facs, i, j):
    if D is not None:
        return D[i][j]
    _, cx, cy = custs[i]
    _, _, fx, fy = facs[j]
    return euclid(cx, cy, fx, fy)


def penalty(custs, facs, i, j):
    demand, _, _ = custs[i]
    setup, cap, _, _ = facs[j]
    return setup * (demand / max(cap, demand, EPS))


def greedy_init(facs, custs, cand, D=None, rng=None):
    n, m = len(facs), len(custs)
    rem = [facs[j][1] for j in range(n)]
    open_ = [False] * n
    assign = [-1] * m
    obj = 0.0

    order = list(range(m))
    if rng is None:
        order.sort(key=lambda i: (-custs[i][0], cand[i][0][0]))
    else:
        order.sort(
            key=lambda i: -custs[i][0] + rng.gauss(0, max(custs[i][0] * 0.05, EPS))
        )

    for i in order:
        demand, _, _ = custs[i]
        best_s, best_j = float("inf"), -1
        for d, j in cand[i]:
            if rem[j] + EPS >= demand:
                s = d + (0.0 if open_[j] else penalty(custs, facs, i, j))
                if s < best_s:
                    best_s, best_j = s, j
        if best_j == -1:
            for j in range(n):
                if rem[j] + EPS >= demand:
                    s = dist(D, custs, facs, i, j) + (
                        0.0 if open_[j] else penalty(custs, facs, i, j)
                    )
                    if s < best_s:
                        best_s, best_j = s, j
        if best_j == -1:
            continue
        if not open_[best_j]:
            open_[best_j] = True
            obj += facs[best_j][0]
        assign[i] = best_j
        rem[best_j] -= demand
        obj += dist(D, custs, facs, i, best_j)

    return assign, obj


def regret_init(facs, custs, cand, inv, D=None):
    n, m = len(facs), len(custs)
    rem = [facs[j][1] for j in range(n)]
    open_ = [False] * n
    assign = [-1] * m
    users = [set() for _ in range(n)]
    obj = 0.0
    done = [False] * m

    def best2(i):
        demand = custs[i][0]
        s1, j1, s2 = float("inf"), -1, float("inf")
        for d, j in cand[i]:
            if rem[j] + EPS >= demand:
                s = d + (0.0 if open_[j] else penalty(custs, facs, i, j))
                if s < s1:
                    s2, s1, j1 = s1, s, j
                elif s < s2:
                    s2 = s
        if j1 == -1:
            for j in range(n):
                if rem[j] + EPS >= demand:
                    s = dist(D, custs, facs, i, j) + (
                        0.0 if open_[j] else penalty(custs, facs, i, j)
                    )
                    if s < s1:
                        s2, s1, j1 = s1, s, j
                    elif s < s2:
                        s2 = s
        r = (s2 - s1) if (j1 != -1 and s2 < float("inf")) else (s1 if j1 != -1 else 0.0)
        return s1, j1, r

    ver = [0] * m
    heap = []
    for i in range(m):
        _, _, r = best2(i)
        heapq.heappush(heap, (-r, i, 0))

    while True:
        ci = cj = -1
        while heap:
            _, i, v = heapq.heappop(heap)
            if done[i] or v < ver[i]:
                continue
            _, j1, _ = best2(i)
            ci, cj = i, j1
            break
        if ci == -1:
            break
        if cj == -1:
            continue
        demand = custs[ci][0]
        if not open_[cj]:
            open_[cj] = True
            obj += facs[cj][0]
        assign[ci] = cj
        users[cj].add(ci)
        rem[cj] -= demand
        obj += dist(D, custs, facs, ci, cj)
        done[ci] = True
        for i2, _ in inv[cj]:
            if not done[i2]:
                ver[i2] += 1
                _, _, r2 = best2(i2)
                heapq.heappush(heap, (-r2, i2, ver[i2]))

    return assign, users, rem, open_, obj


def rebuild(facs, custs, assign, D=None):
    n = len(facs)
    rem = [facs[j][1] for j in range(n)]
    open_ = [False] * n
    users = [set() for _ in range(n)]
    obj = 0.0
    for i, j in enumerate(assign):
        demand, _, _ = custs[i]
        if not open_[j]:
            open_[j] = True
            obj += facs[j][0]
        rem[j] -= demand
        users[j].add(i)
        obj += dist(D, custs, facs, i, j)
    return list(assign), users, rem, open_, obj


def move(assign, users, rem, open_, obj, custs, facs, D, i, new_j):
    old_j = assign[i]
    if old_j == new_j:
        return obj
    demand, _, _ = custs[i]
    old_d = dist(D, custs, facs, i, old_j)
    new_d = dist(D, custs, facs, i, new_j)
    users[old_j].discard(i)
    rem[old_j] += demand
    obj -= old_d
    if not open_[new_j]:
        open_[new_j] = True
        obj += facs[new_j][0]
    assign[i] = new_j
    users[new_j].add(i)
    rem[new_j] -= demand
    obj += new_d
    if not users[old_j] and open_[old_j]:
        open_[old_j] = False
        obj -= facs[old_j][0]
    return obj


def pass_reassign(facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim):
    improved = False
    n, m = len(facs), len(custs)
    order = sorted(
        range(m), key=lambda i: dist(D, custs, facs, i, assign[i]), reverse=True
    )
    for i in order:
        if time.time() - t0 > tlim:
            break
        old_j = assign[i]
        demand, _, _ = custs[i]
        old_d = dist(D, custs, facs, i, old_j)
        save = facs[old_j][0] if len(users[old_j]) == 1 else 0.0
        best_g, best_j, tried = 0.0, old_j, set()
        for d, j in cand[i]:
            tried.add(j)
            if j == old_j or rem[j] + EPS < demand:
                continue
            g = old_d - d + save - (0.0 if open_[j] else facs[j][0])
            if g > best_g + 1e-12:
                best_g, best_j = g, j
        if best_j == old_j:
            for j in range(n):
                if j == old_j or j in tried or not open_[j] or rem[j] + EPS < demand:
                    continue
                g = old_d - dist(D, custs, facs, i, j) + save
                if g > best_g + 1e-12:
                    best_g, best_j = g, j
        if best_j != old_j:
            obj = move(assign, users, rem, open_, obj, custs, facs, D, i, best_j)
            improved = True
    return improved, obj


def pass_swap(facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim):
    improved = False
    m = len(custs)
    order = sorted(
        range(m), key=lambda i: dist(D, custs, facs, i, assign[i]), reverse=True
    )
    for i1 in order:
        if time.time() - t0 > tlim:
            break
        j1 = assign[i1]
        demand1, _, _ = custs[i1]
        d1_j1 = dist(D, custs, facs, i1, j1)
        best_g, best_i2, best_j2, best_d1j2, best_d2j1 = 1e-12, -1, -1, 0.0, 0.0
        for d1_j2, j2 in cand[i1]:
            if j2 == j1 or not open_[j2]:
                continue
            for i2 in list(users[j2]):
                demand2, _, _ = custs[i2]
                if rem[j2] + demand2 - demand1 + EPS < 0:
                    continue
                if rem[j1] + demand1 - demand2 + EPS < 0:
                    continue
                d2_j2 = dist(D, custs, facs, i2, j2)
                d2_j1 = dist(D, custs, facs, i2, j1)
                g = (d1_j1 - d1_j2) + (d2_j2 - d2_j1)
                if g > best_g:
                    best_g, best_i2, best_j2, best_d1j2, best_d2j1 = (
                        g,
                        i2,
                        j2,
                        d1_j2,
                        d2_j1,
                    )
        if best_i2 != -1:
            i2, j2 = best_i2, best_j2
            demand2, _, _ = custs[i2]
            d2_j2 = dist(D, custs, facs, i2, j2)
            users[j1].discard(i1)
            users[j2].discard(i2)
            rem[j1] += demand1 - demand2
            rem[j2] += demand2 - demand1
            assign[i1] = j2
            assign[i2] = j1
            users[j2].add(i1)
            users[j1].add(i2)
            obj = obj - d1_j1 - d2_j2 + best_d1j2 + best_d2j1
            improved = True
    return improved, obj


def pass_close(facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim):
    improved = False
    n = len(facs)
    for j in sorted(
        [j for j in range(n) if open_[j] and users[j]],
        key=lambda j: (len(users[j]), facs[j][0]),
    ):
        if time.time() - t0 > tlim:
            break
        custs_j = sorted(users[j], key=lambda i: -custs[i][0])
        rem_c = rem[:]
        for i in custs_j:
            rem_c[j] += custs[i][0]
        delta, reas, ok = facs[j][0], [], True
        for i in custs_j:
            demand, _, _ = custs[i]
            old_d = dist(D, custs, facs, i, j)
            best_k, best_d, tried = -1, float("inf"), set()
            for d, k in cand[i]:
                tried.add(k)
                if k == j or not open_[k]:
                    continue
                if rem_c[k] + EPS >= demand and d < best_d:
                    best_d, best_k = d, k
            if best_k == -1:
                for k in range(n):
                    if k == j or not open_[k] or k in tried or rem_c[k] + EPS < demand:
                        continue
                    d = dist(D, custs, facs, i, k)
                    if d < best_d:
                        best_d, best_k = d, k
            if best_k == -1:
                ok = False
                break
            rem_c[best_k] -= demand
            delta += old_d - best_d
            reas.append((i, best_k))
        if ok and delta > 1e-12:
            for i, nj in reas:
                obj = move(assign, users, rem, open_, obj, custs, facs, D, i, nj)
            improved = True
    return improved, obj


def pass_substitute(facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim):
    improved = False
    n = len(facs)
    for j1 in sorted(
        [j for j in range(n) if open_[j] and users[j]], key=lambda j: len(users[j])
    ):
        if time.time() - t0 > tlim:
            break
        setup1 = facs[j1][0]
        custs_j1 = sorted(list(users[j1]), key=lambda i: -custs[i][0])
        nearby = {j for i in custs_j1 for _, j in cand[i] if not open_[j]}
        best_g, best_j2, best_reas = 1e-12, -1, []
        for j2 in nearby:
            setup2, cap2, _, _ = facs[j2]
            rem_c = rem[:]
            rem_c[j2] = cap2
            delta = setup1 - setup2
            reas, ok = [], True
            for i in custs_j1:
                demand, _, _ = custs[i]
                old_d = dist(D, custs, facs, i, j1)
                best_k, best_d, tried = -1, float("inf"), set()
                for d, k in cand[i]:
                    tried.add(k)
                    if k == j1:
                        continue
                    if (
                        (open_[k] or k == j2)
                        and rem_c[k] + EPS >= demand
                        and d < best_d
                    ):
                        best_d, best_k = d, k
                if best_k == -1:
                    for k in range(n):
                        if k == j1 or k in tried or not (open_[k] or k == j2):
                            continue
                        if rem_c[k] + EPS < demand:
                            continue
                        d = dist(D, custs, facs, i, k)
                        if d < best_d:
                            best_d, best_k = d, k
                if best_k == -1:
                    ok = False
                    break
                rem_c[best_k] -= demand
                delta += old_d - best_d
                reas.append((i, best_k))
            if ok and delta > best_g:
                best_g, best_j2, best_reas = delta, j2, reas
        if best_j2 != -1:
            if not open_[best_j2]:
                open_[best_j2] = True
                obj += facs[best_j2][0]
            for i, nk in best_reas:
                obj = move(assign, users, rem, open_, obj, custs, facs, D, i, nk)
            improved = True
    return improved, obj


def pass_open(facs, custs, assign, users, rem, open_, obj, D, inv, t0, tlim):
    improved = False
    n = len(facs)
    scores = []
    for j in range(n):
        if open_[j]:
            continue
        score, cnt = -facs[j][0], 0
        for i, d_j in inv[j]:
            g = dist(D, custs, facs, i, assign[i]) - d_j
            if g > 0:
                score += g
                cnt += 1
        if cnt > 0:
            scores.append((score, j))
    scores.sort(reverse=True)
    for score, j in scores[: (20 if n <= 200 else 30)]:
        if time.time() - t0 > tlim:
            break
        if score <= 0:
            break
        setup, cap, _, _ = facs[j]
        moves = []
        for i, d_j in inv[j]:
            g = dist(D, custs, facs, i, assign[i]) - d_j
            if g > 0:
                moves.append((g / max(custs[i][0], EPS), g, i, custs[i][0]))
        if not moves:
            continue
        moves.sort(reverse=True)
        chosen, used, total_g = [], 0.0, -setup
        for _, g, i, demand in moves:
            if used + demand <= cap + EPS:
                chosen.append(i)
                used += demand
                total_g += g
        if total_g <= 1e-12:
            continue
        for i in chosen:
            obj = move(assign, users, rem, open_, obj, custs, facs, D, i, j)
        improved = True
    return improved, obj


def perturb(facs, custs, cand, assign, users, rem, open_, obj, D, rng, n_close):
    open_list = [j for j in range(len(facs)) if open_[j] and users[j]]
    if len(open_list) <= n_close:
        return obj
    scored = [(len(users[j]) + rng.uniform(0, 2), j) for j in open_list]
    scored.sort()
    to_close = [j for _, j in scored[:n_close]]
    displaced = [i for j in to_close for i in list(users[j])]
    for i in displaced:
        old_j = assign[i]
        demand, _, _ = custs[i]
        users[old_j].discard(i)
        rem[old_j] += demand
        obj -= dist(D, custs, facs, i, old_j)
        assign[i] = -1
    for j in to_close:
        if not users[j] and open_[j]:
            open_[j] = False
            obj -= facs[j][0]
    rng.shuffle(displaced)
    for i in displaced:
        demand, _, _ = custs[i]
        best_s, best_j = float("inf"), -1
        for d, j in cand[i]:
            if rem[j] + EPS >= demand:
                s = d + (0.0 if open_[j] else penalty(custs, facs, i, j))
                if s < best_s:
                    best_s, best_j = s, j
        if best_j == -1:
            for j in range(len(facs)):
                if rem[j] + EPS >= demand:
                    s = dist(D, custs, facs, i, j) + (
                        0.0 if open_[j] else penalty(custs, facs, i, j)
                    )
                    if s < best_s:
                        best_s, best_j = s, j
        if best_j == -1:
            continue
        if not open_[best_j]:
            open_[best_j] = True
            obj += facs[best_j][0]
        assign[i] = best_j
        users[best_j].add(i)
        rem[best_j] -= demand
        obj += dist(D, custs, facs, i, best_j)
    return obj


def solve(facs, custs):
    n, m = len(facs), len(custs)
    rng = random.Random(42)
    tlim = TLIM
    t0 = time.time()

    D = build_dist(facs, custs) if n * m <= 5_000_000 else None
    cand = build_cands(facs, custs, 50)
    inv = build_inv(cand, n)

    assign, users, rem, open_, obj = regret_init(facs, custs, cand, inv, D)
    best_obj, best_assign = obj, assign[:]

    init_budget = min(tlim * 0.1, 3.0)
    while time.time() - t0 < init_budget:
        a, v = greedy_init(facs, custs, cand, D, rng=rng)
        if v < best_obj:
            best_obj, best_assign = v, a[:]

    assign, users, rem, open_, obj = rebuild(facs, custs, best_assign, D)
    best_obj, best_assign = obj, assign[:]

    stagnation = 0
    while time.time() - t0 <= tlim:
        changed = False
        for fn in [
            lambda: pass_reassign(
                facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim
            ),
            lambda: pass_swap(
                facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim
            ),
            lambda: pass_close(
                facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim
            ),
            lambda: pass_substitute(
                facs, custs, cand, assign, users, rem, open_, obj, D, t0, tlim
            ),
            lambda: pass_open(
                facs, custs, assign, users, rem, open_, obj, D, inv, t0, tlim
            ),
        ]:
            if time.time() - t0 > tlim:
                break
            imp, obj = fn()
            changed |= imp

        if obj < best_obj:
            best_obj, best_assign = obj, assign[:]

        if not changed:
            stagnation += 1
            if stagnation >= 6:
                stagnation = 0
                assign, users, rem, open_, obj = rebuild(facs, custs, best_assign, D)
                n_open = sum(1 for j in range(n) if open_[j])
                n_close = rng.randint(1, max(1, min(8, n_open // 3)))
                obj = perturb(
                    facs, custs, cand, assign, users, rem, open_, obj, D, rng, n_close
                )
        else:
            stagnation = 0

    if best_obj < obj:
        assign = best_assign
    return sum(facs[j][0] for j in set(assign)) + sum(
        dist(D, custs, facs, i, j) for i, j in enumerate(assign)
    )


def main():
    tokens = sys.stdin.read().strip().split()
    facs, custs = read_input(tokens)
    print(f"{solve(facs, custs):.6f}")


if __name__ == "__main__":
    main()

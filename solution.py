from pathlib import Path
import random
import sys
import time


def parse_test(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    n, m = map(int, lines[0].split())

    costs = []
    sets = []
    elem_to_sets = [[] for _ in range(n)]

    for i in range(1, m + 1):
        parts = list(map(int, lines[i].split()))
        c, elems = parts[0], parts[1:]
        costs.append(c)
        sets.append(elems)
        for e in elems:
            elem_to_sets[e].append(i - 1)

    return n, m, costs, sets, elem_to_sets


def build_cover_count(n, sets, chosen):
    cover_count = [0] * n
    for i, v in enumerate(chosen):
        if v:
            for e in sets[i]:
                cover_count[e] += 1
    return cover_count


def objective(costs, chosen):
    return sum(costs[i] for i, v in enumerate(chosen) if v)


def reverse_delete(n, costs, sets, chosen, cover_count=None):
    if cover_count is None:
        cover_count = build_cover_count(n, sets, chosen)

    selected = [i for i, v in enumerate(chosen) if v]
    selected.sort(key=lambda i: (costs[i], len(sets[i])), reverse=True)

    for i in selected:
        if all(cover_count[e] > 1 for e in sets[i]):
            chosen[i] = 0
            for e in sets[i]:
                cover_count[e] -= 1

    return objective(costs, chosen), chosen, cover_count


def greedy_randomized(n, m, costs, sets, elem_to_sets, alpha, seed):
    rng = random.Random(seed)

    deg = [len(elem_to_sets[e]) for e in range(n)]

    rarity = [1.0 / d for d in deg]
    chosen = [0] * m
    uncovered = [1] * n
    uncovered_count = n

    gain_count = [len(s) for s in sets]
    gain_rarity = [sum(rarity[e] for e in s) for s in sets]

    forced_sets = set()
    for e in range(n):
        if deg[e] == 1:
            forced_sets.add(elem_to_sets[e][0])

    for sidx in forced_sets:
        if chosen[sidx]:
            continue
        chosen[sidx] = 1
        for e in sets[sidx]:
            if uncovered[e]:
                uncovered[e] = 0
                uncovered_count -= 1
                r = rarity[e]
                for t in elem_to_sets[e]:
                    if not chosen[t] and gain_count[t] > 0:
                        gain_count[t] -= 1
                        gain_rarity[t] -= r

    if uncovered_count == 0:
        obj, chosen, _ = reverse_delete(n, costs, sets, chosen)
        return obj, chosen

    degree_range = max(deg) - min(deg)
    rarity_weight = 1.5 if degree_range > 10 else 1.0

    while uncovered_count > 0:
        best_score = float("inf")
        best_i = -1

        for i in range(m):
            if chosen[i] or gain_count[i] <= 0:
                continue
            weighted_gain = gain_count[i] + rarity_weight * gain_rarity[i]
            score = costs[i] / weighted_gain
            if score < best_score:
                best_score = score
                best_i = i

        pick = best_i
        if alpha > 0.0:
            limit = best_score * (1.0 + alpha)
            rcl = []
            for i in range(m):
                if chosen[i] or gain_count[i] <= 0:
                    continue
                weighted_gain = gain_count[i] + rarity_weight * gain_rarity[i]
                score = costs[i] / weighted_gain
                if score <= limit:
                    rcl.append(i)
            if rcl:
                pick = rng.choice(rcl)

        chosen[pick] = 1
        for e in sets[pick]:
            if uncovered[e]:
                uncovered[e] = 0
                uncovered_count -= 1
                r = rarity[e]
                for t in elem_to_sets[e]:
                    if not chosen[t] and gain_count[t] > 0:
                        gain_count[t] -= 1
                        gain_rarity[t] -= r

    obj, chosen, _ = reverse_delete(n, costs, sets, chosen)
    return obj, chosen


def one_swap_improve(costs, sets, elem_to_sets, chosen, cover_count):
    selected = [i for i, v in enumerate(chosen) if v]
    selected.sort(key=lambda i: costs[i], reverse=True)

    for i in selected:
        uniques = [e for e in sets[i] if cover_count[e] == 1]
        if not uniques:
            continue

        freq = {}
        need = len(uniques)
        for e in uniques:
            for j in elem_to_sets[e]:
                if j == i or chosen[j]:
                    continue
                freq[j] = freq.get(j, 0) + 1

        best_j = -1
        best_cost = costs[i]
        for j, cnt in freq.items():
            if cnt == need and costs[j] < best_cost:
                best_cost = costs[j]
                best_j = j

        if best_j == -1:
            continue

        chosen[i] = 0
        for e in sets[i]:
            cover_count[e] -= 1
        chosen[best_j] = 1
        for e in sets[best_j]:
            cover_count[e] += 1
        return True

    return False


def add_drop_improve(n, costs, sets, chosen, cover_count):
    unselected = [i for i in range(len(sets)) if not chosen[i]]
    if not unselected:
        return False

    scored = []
    for j in unselected:
        uniq_hit = 0
        for e in sets[j]:
            if cover_count[e] == 1:
                uniq_hit += 1
        if uniq_hit == 0:
            continue
        score = costs[j] / uniq_hit
        scored.append((score, j))

    if not scored:
        return False

    scored.sort()
    base_obj = objective(costs, chosen)

    for _, j in scored[:30]:
        chosen[j] = 1
        for e in sets[j]:
            cover_count[e] += 1

        cur_obj, chosen, cover_count = reverse_delete(
            n, costs, sets, chosen, cover_count
        )
        if cur_obj < base_obj:
            return True

        if chosen[j]:
            chosen[j] = 0
            for e in sets[j]:
                cover_count[e] -= 1

    return False


def local_search(n, costs, sets, elem_to_sets, chosen):
    t_end = time.time() + 1.5
    cover_count = build_cover_count(n, sets, chosen)

    improved = True
    while improved and time.time() < t_end:
        improved = False

        _, chosen, cover_count = reverse_delete(n, costs, sets, chosen, cover_count)

        if one_swap_improve(costs, sets, elem_to_sets, chosen, cover_count):
            improved = True
            continue

        if add_drop_improve(n, costs, sets, chosen, cover_count):
            improved = True

    obj, chosen, _ = reverse_delete(n, costs, sets, chosen, cover_count)
    return obj, chosen


def multistart_plan_by_size(m):
    starts = [(0.0, 1)]
    if m <= 400:
        starts += [(0.10, 60), (0.25, 90)]
    elif m <= 2000:
        starts += [(0.10, 30), (0.25, 45)]
    else:
        starts += [(0.10, 10), (0.20, 15)]
    return starts


def solve_test_case(path):
    n, m, costs, sets, elem_to_sets = parse_test(path)

    starts = multistart_plan_by_size(m)

    best_obj = float("inf")
    best_sol = None

    for alpha, iters in starts:
        for k in range(iters):
            seed = 1234567 + 10007 * k + int(alpha * 1000)
            obj, sol = greedy_randomized(
                n, m, costs, sets, elem_to_sets, alpha=alpha, seed=seed
            )
            if obj < best_obj:
                best_obj = obj
                best_sol = sol

    best_obj, best_sol = local_search(n, costs, sets, elem_to_sets, best_sol)
    return best_obj, best_sol


if __name__ == "__main__":
    test_name = sys.argv[1]
    instance_path = Path("Setcover") / test_name
    obj, solution = solve_test_case(instance_path)
    print(f"{obj} 0\n" + " ".join(map(str, solution)))

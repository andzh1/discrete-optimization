#!/usr/bin/env python3
import sys
import time
import random


def solve(input_file):
    start = time.time()
    TIME_LIMIT = 60
    
    data = open(input_file).read().split()
    n, m = int(data[0]), int(data[1])
    adj = [[] for _ in range(n)]
    for i in range(m):
        u, v = int(data[2 + 2 * i]), int(data[3 + 2 * i])
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    adj = [list(set(a)) for a in adj]
    deg = [len(adj[v]) for v in range(n)]

    def dsatur():
        colors = [-1] * n
        sat = [0] * n
        nbr_cols = [set() for _ in range(n)]
        uncolored = set(range(n))
        for _ in range(n):
            v = max(uncolored, key=lambda x: (sat[x], deg[x]))
            uncolored.remove(v)
            c = 0
            while c in nbr_cols[v]:
                c += 1
            colors[v] = c
            for u in adj[v]:
                if colors[u] == -1 and c not in nbr_cols[u]:
                    nbr_cols[u].add(c)
                    sat[u] += 1
        return colors

    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        colors = [-1] * n
        nbr_cols = [set() for _ in range(n)]
        for v in order:
            c = 0
            while c in nbr_cols[v]:
                c += 1
            colors[v] = c
            for u in adj[v]:
                nbr_cols[u].add(c)
        return colors

    def tabucol(k, budget, init=None):
        if k < 2:
            return ([0] * n) if (m == 0) else None
        end_t = time.time() + budget

        if init is not None:
            col = [c % k for c in init]
        else:
            col = [random.randint(0, k - 1) for _ in range(n)]

        cc = [[0] * k for _ in range(n)]
        for v in range(n):
            for u in adj[v]:
                cc[v][col[u]] += 1

        tabu = [[0] * k for _ in range(n)]
        tenure_base = max(5, k // 4)
        itr = 0
        max_iter = max(10000, n * k * 2)

        while itr < max_iter and time.time() < end_t:
            conf = [v for v in range(n) if cc[v][col[v]] > 0]
            if not conf:
                return col 
            
            sample = conf if len(conf) <= 40 else random.sample(conf, 40)

            best_d = float('inf')
            bv = bc = -1

            for v in sample:
                curr_c = col[v]
                curr_cost = cc[v][curr_c]
                for c in range(k):
                    if c == curr_c:
                        continue
                    d = cc[v][c] - curr_cost
                    if tabu[v][c] > itr:
                        if not (d < 0 and cc[v][c] == 0):
                            continue
                    if d < best_d:
                        best_d = d
                        bv, bc = v, c

            if bv == -1:
                bv = random.choice(conf)
                bc = random.randint(0, k - 1)
                while bc == col[bv]:
                    bc = random.randint(0, k - 1)

            old_c = col[bv]
            tabu[bv][old_c] = itr + tenure_base + random.randint(0, tenure_base)
            for u in adj[bv]:
                cc[u][old_c] -= 1
                cc[u][bc] += 1
            col[bv] = bc
            itr += 1

        return None
    
    def greedy_reduce(colors):
        k = max(colors) + 1
        target = k - 1
        col = list(colors)

        verts = sorted([v for v in range(n) if col[v] == target],
                       key=lambda x: -deg[x])
        for v in verts:
            nbr_c = {col[u] for u in adj[v]}
            alt = next((c for c in range(target) if c not in nbr_c), None)
            if alt is None:
                return None
            col[v] = alt
        return col

    adj_set = [set(a) for a in adj]

    def greedy_clique_lb():
        clique = []
        for v in sorted(range(n), key=lambda x: -deg[x]):
            if all(v in adj_set[u] for u in clique):
                clique.append(v)
        return max(1, len(clique))

    lb = greedy_clique_lb()

    best = dsatur()
    best_k = max(best) + 1
    print(f"DSatur: {best_k} colors", file=sys.stderr)

    greedy_budget = min(20.0, TIME_LIMIT * 0.05)
    for seed in range(10000):
        if time.time() - start > greedy_budget:
            break
        c = greedy_random(seed)
        k = max(c) + 1
        if k < best_k:
            best, best_k = c, k
            print(f"  RandGreedy seed={seed}: {k}", file=sys.stderr)

    print(f"After random greedy: {best_k} colors", file=sys.stderr)

    while True:
        r = greedy_reduce(best)
        if r is None:
            break
        best = r
        best_k = max(best) + 1
        print(f"  GreedyElim: {best_k}", file=sys.stderr)

    print(f"After greedy elim: {best_k} colors", file=sys.stderr)

    k = best_k - 1
    while k >= lb and time.time() - start < TIME_LIMIT:
        remaining = TIME_LIMIT - (time.time() - start)
        budget_per_k = min(remaining * 0.5, 60.0)
        print(f"TabuCol k={k}, budget={budget_per_k:.1f}s", file=sys.stderr)

        result = None
        attempt_end = time.time() + budget_per_k
        seed = 0
        while time.time() < attempt_end:
            sub = min(attempt_end - time.time(), 15.0)
            if sub <= 0:
                break
            random.seed(seed)
            r = tabucol(k, sub, init=best)
            if r is not None:
                result = r
                break
            seed += 1

        if result is not None:
            best, best_k = result, k
            print(f"  → Found k={k}!", file=sys.stderr)
            while True:
                r = greedy_reduce(best)
                if r is None:
                    break
                best = r
                best_k = max(best) + 1
                print(f"  GreedyElim after TabuCol: {best_k}", file=sys.stderr)
            k = best_k - 1
        else:
            print(f"  → Failed k={k}", file=sys.stderr)
            break

    return best_k, best


if __name__ == '__main__':
    k, colors = solve(sys.argv[1])
    print(f"{k} 0")
    print(' '.join(map(str, colors)))

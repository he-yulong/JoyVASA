# run: python tools/project_map.py
import os, ast, sys, pathlib, textwrap, json

ROOT = pathlib.Path(__file__).resolve().parents[1]


def tree(path, prefix=""):
    items = sorted([p for p in path.iterdir() if not p.name.startswith(".")],
                   key=lambda p: (not p.is_dir(), p.name.lower()))
    for i, p in enumerate(items):
        joint = "└── " if i == len(items) - 1 else "├── "
        print(prefix + joint + p.name)
        if p.is_dir():
            tree(p, prefix + ("    " if i == len(items) - 1 else "│   "))


def py_files(root):
    return [p for p in root.rglob("*.py") if "venv" not in str(p) and "build" not in str(p)]


def parse_imports(py):
    mod = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
    imps = []
    for n in ast.walk(mod):
        if isinstance(n, ast.Import):
            imps += [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imps.append(n.module.split(".")[0])
    return sorted(set(imps))


def has_main(py):
    return "if __name__ == \"__main__\"" in py.read_text(encoding="utf-8", errors="ignore")


def main():
    print("## Top-level tree\n")
    tree(ROOT);
    print()
    print("## Entrypoints\n")
    for p in py_files(ROOT):
        if p.name in ("train.py", "inference.py", "eval.py") or has_main(p):
            print("-", p.relative_to(ROOT))
    print("\n## Import graph (module -> direct imports)\n")
    graph = {}
    for p in py_files(ROOT):
        rel = p.relative_to(ROOT)
        graph[str(rel)] = parse_imports(p)
    # show only first-level internal imports
    for k, v in graph.items():
        internal = [i for i in v if (ROOT / i).exists() or any(str(pp).startswith(i) for pp in graph.keys())]
        if internal:
            print(f"{k} -> {internal}")
    # cycles (rough heuristic)
    print("\n## Potential import cycles (heuristic)")
    for a in graph:
        for b in graph:
            if a != b and any(str(b).startswith(i) for i in graph[a]) and any(str(a).startswith(i) for i in graph[b]):
                print(f"- {a} <-> {b}")
    # summary
    print("\n## Summary")
    print(f"Python files: {len(graph)}")


if __name__ == "__main__":
    sys.setrecursionlimit(10 ** 6)
    main()

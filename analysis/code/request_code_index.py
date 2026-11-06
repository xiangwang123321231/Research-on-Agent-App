"""Declaration-verified class references for request-driven graph selection.

The index stores references, not method bodies or LLM signals. APK paths,
class names and their lexical order are never evidence of LLM functionality.
"""
from collections import defaultdict, deque
from pathlib import Path
import re


def select_request_files(paths, seeds, scope, associated=()):
    owners, calls, dependencies = {}, {}, {}
    reverse = defaultdict(set)
    errors = 0
    for path in paths:
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            errors += 1
            continue
        match = re.search(r'(?m)^\s*\.class\b[^\n]*\s(L[^;\s]+;)\s*$', text)
        if not match:
            continue
        owner = match[1]
        if owner in owners:
            # Ambiguous class definitions cannot silently choose a file.
            owners[owner] = None
            errors += 1
        else:
            owners[owner] = path
        invokes = set(re.findall(r'(?m)^\s*invoke-[\w/-]+\s+\{[^}]*\},\s*(L[^;\s]+;)->', text))
        calls[owner] = invokes
        dependencies[owner] = invokes | set(re.findall(
            r'(?m)^\s*(?:\.field|\.method|\.super|\.implements|new-instance|const-class|check-cast|iget\S*|iput\S*)[^\n]*?(L[^;\s<>]+;)', text))
        for target in invokes:
            reverse[target].add(owner)
    by_path = {path: owner for owner, path in owners.items() if path is not None}
    chosen = {by_path[item] if item in by_path else item for item in seeds
              if item in by_path or item in owners and owners[item] is not None}
    pending = deque(sorted(chosen))
    associated = set(associated)
    def relevant(owner):
        return scope.matches_owner(owner) or owner in associated
    while pending:
        owner = pending.popleft()
        # Only callers with established application provenance are admitted.
        neighbours = {c for c in reverse[owner] if relevant(c)}
        neighbours.update(c for c in dependencies.get(owner, ()) if relevant(c))
        for target in sorted(neighbours-chosen):
            if owners.get(target) is not None:
                chosen.add(target)
                pending.append(target)
    return sorted(owners[owner] for owner in chosen if owners.get(owner) is not None), errors

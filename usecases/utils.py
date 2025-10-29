from pathlib import Path

# Setup paths
def get_repo_root():
    """Get repository root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    return Path.cwd()

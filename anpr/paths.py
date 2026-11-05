"""Reject output names that would replace source assets or another output role."""

from pathlib import Path


def _path_identity(path):
    """Return a canonical path and any existing filesystem identity, without opening data."""
    # Resolved paths catch symlinks and spelling differences, including nonexistent destinations.
    resolved_path = Path(path).resolve()
    try:
        # Device and inode identity also catches hard links with unrelated visible filenames.
        file_stat = resolved_path.stat()
    except FileNotFoundError:
        # A fresh output has no inode yet, but its canonical name is still protected.
        return resolved_path, None
    return resolved_path, (file_stat.st_dev, file_stat.st_ino)


def protect_output_paths(inputs, outputs):
    """Raise ValueError when an output aliases an input or another output.

    Inputs and outputs are iterables of local path strings or Paths. Existing
    distinct outputs remain replaceable for normal reruns. This validation reads
    filesystem metadata only; callers must run it before opening any output.
    It is an input-mistake guard, not protection against concurrent path changes.
    """
    # Index source ownership once so checking a large crop manifest costs O(inputs + outputs).
    input_paths = set()
    input_inodes = set()
    for supplied_path in inputs:
        # Sources may not exist yet; name ownership still prevents destructive command configurations.
        resolved_path, inode = _path_identity(supplied_path)
        input_paths.add(resolved_path)
        if inode is not None:
            input_inodes.add(inode)
    # Output roles must also remain distinct from one another, such as report versus calibration.
    output_paths = set()
    output_inodes = set()
    for supplied_path in outputs:
        # Check physical aliases before any caller truncates or replaces its selected destination.
        resolved_path, inode = _path_identity(supplied_path)
        if resolved_path in input_paths or (inode is not None and inode in input_inodes):
            raise ValueError(f"Output aliases a protected input: {supplied_path}")
        if resolved_path in output_paths or (inode is not None and inode in output_inodes):
            raise ValueError(f"Output roles alias each other: {supplied_path}")
        output_paths.add(resolved_path)
        if inode is not None:
            output_inodes.add(inode)

"""Paths to course directories in this workspace."""

from pathlib import Path


class PathManager:
    """Locate course data independently of the working directory."""

    ROOT = Path.home()
    DESKTOP = ROOT / "Desktop"
    BASE = Path(__file__).resolve().parent.parent

    FIGURES = BASE / "figures"
    DATA = BASE / "data"

    @classmethod
    def setup_directories(cls):
        """Check that all course directories exist."""
        for course in (
            cls.MEDICAL_PHYSICS,
            cls.MATERIAL_PHYSICS,
            cls.FUNCTIONAL_MATERIALS,
        ):
            if not course.is_dir():
                raise RuntimeError(f"Course {course.name} does not exist at {course}.")

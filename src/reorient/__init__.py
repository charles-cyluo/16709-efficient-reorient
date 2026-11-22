"""Learning efficient in-hand reorientation using gravity (MuJoCo + RL)."""
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
SCENE_XML = ASSETS_DIR / "scene.xml"

TABLE_Z = 0.72  # table-top height [m]; keep in sync with assets/scene.xml

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

import robocasa
from robocasa.models.scenes.scene_builder import create_tabletop_fixtures


# base class for tabletop scenes
class TabletopArena(Arena):
    """
    Tabletop arena class holding all of the fixtures

    Args:
        layout_id (int or TabletopLayoutType): layout of the tabletop to load

        style_id (int or StyleType): style of the tabletop to load

        rng (np.random.Generator): random number generator used for initializing
            fixture state in the TabletopArena
    """

    def __init__(self, layout_id, style_id, rng=None):
        super().__init__(
            xml_path_completion(
                "arenas/empty_tabletop_arena.xml", root=robocasa.models.assets_root
            )
        )
        self.fixtures = create_tabletop_fixtures(
            layout_id=layout_id,
            style_id=style_id,
            rng=rng,
        )

    def get_fixture_cfgs(self):
        """
        Returns config data for all fixtures in the arena

        Returns:
            list: list of fixture configurations
        """
        fixture_cfgs = []
        for name, fxtr in self.fixtures.items():
            cfg = {}
            cfg["name"] = name
            cfg["model"] = fxtr
            cfg["type"] = "fixture"
            if hasattr(fxtr, "_placement"):
                cfg["placement"] = fxtr._placement

            fixture_cfgs.append(cfg)

        return fixture_cfgs

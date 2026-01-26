"""Launch module for Beaker job submission and management."""

from olmix.launch.launch_utils import (
    config_from_path,
    mk_mixes,
    mk_source_instances,
    prettify_mixes,
)

__all__ = [
    "config_from_path",
    "mk_mixes",
    "mk_source_instances",
    "prettify_mixes",
]

# Beaker functions are optional - only available if beaker-py is installed
try:
    from olmix.launch.beaker import (
        get_beaker_username,
        mk_experiment_group,
        mk_experiments,
        mk_instance_cmd,
        mk_launch_configs,
    )

    __all__.extend(
        [
            "get_beaker_username",
            "mk_experiment_group",
            "mk_experiments",
            "mk_instance_cmd",
            "mk_launch_configs",
        ]
    )
except ImportError:
    pass

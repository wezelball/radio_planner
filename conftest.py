"""
conftest.py
Workaround for ROS/ament pytest plugins conflicting with pytest >= 8.
These plugins register hooks that are incompatible with newer pytest versions.
This file deregisters them before test collection begins.
"""

def pytest_configure(config):
    """Unregister incompatible ROS pytest plugins if present."""
    ros_plugins = [
        "launch_testing_ros_pytest_entrypoint",
        "launch_testing",
        "ament_lint",
        "ament_copyright",
        "ament_flake8",
        "ament_xmllint",
        "ament_pep257",
    ]
    pm = config.pluginmanager
    for name in ros_plugins:
        plugin = pm.get_plugin(name)
        if plugin is not None:
            pm.unregister(plugin)

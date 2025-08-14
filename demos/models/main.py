import orca
from config import get_config, DEMOSConfig

@orca.injectable("year")
def year():
    default_year = get_config().base_year
    iter_var = orca.get_injectable("iter_var")
    if iter_var is not None:
        return iter_var
    else:
        return default_year

demos_config: DEMOSConfig = get_config()
orca.add_injectable("sim_steps", demos_config.modules)

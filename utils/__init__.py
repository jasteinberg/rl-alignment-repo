"""
utils/__init__.py
=================
Convenience re-exports. In notebooks:

    import sys; sys.path.insert(0, ENV.REPO)
    from utils.env import ENV
    from utils.reward_model import BradleyTerry, make_synthetic_preferences
    from utils.toy_mdp import GridWorldEnv, TabularPolicy, reinforce
    from utils.train import plot_history
"""

from .env import ENV

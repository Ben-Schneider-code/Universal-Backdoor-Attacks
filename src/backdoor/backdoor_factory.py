from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.poison.clean_label.adv_clean_label import AdversarialCleanLabel
from src.backdoor.backdoor import Backdoor
from src.backdoor.poison.clean_label.badnet_clean import BadnetClean
from src.backdoor.poison.clean_label.no_backdoor import NoBackdoor
from src.backdoor.poison.clean_label.wanet import Wanet
from src.backdoor.poison.poison_label.adaptive_blend import AdaptiveBlend
from src.backdoor.poison.poison_label.adaptive_patch import AdaptivePatch
from src.backdoor.poison.poison_label.badnet import Badnet
from src.backdoor.poison.clean_label.refool import Refool
from src.backdoor.poison.poison_label.universal_backdoor import UniversalBackdoor, UBA
from src.backdoor.poison.poison_label.many_trigger_badnet import ManyTriggerBadnet
from src.backdoor.supply_chain.latent_backdoor import LatentBackdoor
from src.backdoor.supply_chain.parameter_controlled_backdoor import ParameterControlledBackdoor
from src.backdoor.poison.poison_label.multi_badnets import MultiBadnets

class BackdoorFactory:
    @staticmethod
    def from_backdoor_args(backdoor_args: BackdoorArgs, env_args: EnvArgs = None) -> Backdoor:
        if backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.badnet:
            return Badnet(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.refool:
            return Refool(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.adv_clean:
            return AdversarialCleanLabel(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.adaptive_blend:
            return AdaptiveBlend(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.adaptive_patch:
            return AdaptivePatch(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.wanet:
            return Wanet(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.latent_backdoor:
            return LatentBackdoor(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.no_backdoor:
            return NoBackdoor(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.badnet_clean:
            return BadnetClean(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.many_trigger_badnet:
            return ManyTriggerBadnet(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.handcrafted:
            return ParameterControlledBackdoor(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.binary_map_poison:
            return UBA(backdoor_args, env_args=env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.multi_badnets:
            return MultiBadnets(backdoor_args, env_args)
        elif backdoor_args.backdoor_name == BackdoorArgs.ATTACKS.universal_backdoor:
            return UniversalBackdoor(backdoor_args, env_args)
        else:
            raise ValueError(backdoor_args.backdoor_name)
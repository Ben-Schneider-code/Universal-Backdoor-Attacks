from src.arguments.env_args import EnvArgs
from src.arguments.grid_evaluate_args import GridEvaluateArgs
from src.arguments.grid_search_args import GridSearchArgs
from src.arguments.observer_args import ObserverArgs
from src.observers.before_deployment_robustness_observer import BeforeDeploymentRobustnessObserver
from src.observers.grid_evaluate_observer import GridEvaluateObserver
from src.observers.grid_search_observer import GridSearchObserver


class ObserverFactory:

    @staticmethod
    def from_observer_args(observer_args: ObserverArgs, grid_search_args: GridSearchArgs = None,
                           grid_evaluate_args: GridEvaluateArgs = None,
                           env_args: EnvArgs = None):
        observers = []
        for observer_name in observer_args.observer_names:
            if observer_name.lower() == BeforeDeploymentRobustnessObserver.KEY:
                observers += [BeforeDeploymentRobustnessObserver(observer_args, env_args=env_args)]
            elif observer_name.lower() == GridSearchObserver.KEY:
                observers += [GridSearchObserver(observer_args, grid_search_args=grid_search_args, env_args=env_args)]
            elif observer_name.lower() == GridEvaluateObserver.KEY:
                observers += [GridEvaluateObserver(observer_args, grid_evaluate_args=grid_evaluate_args, env_args=env_args)]
            else:
                raise ValueError
        return observers
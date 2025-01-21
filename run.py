import argparse
import os

def _discover_experiments():
    """Returns all available experiments"""
    return [obj.name for obj in os.scandir("experiments") if obj.is_dir()]

def run_experiment(args):
    experiment_name: str = args.experiment_name

    import importlib
    print(f"Running experiment: '{experiment_name}'")
    experiment_module = importlib.import_module(f"experiments.{experiment_name}.run")
    experiment_module.run(args)
    print(f"Finished experiment: '{experiment_name}'")


def main():
    parser = argparse.ArgumentParser(description="Run different tasks or experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-parser for the "experiment" command
    experiment_parser = subparsers.add_parser("experiment", help="Run an experiment")
    experiment_parser.add_argument(
        "experiment_name",
        choices=_discover_experiments(),
        help="The name of the experiment to run"
    )
    experiment_parser.set_defaults(func=lambda args: run_experiment(args))

    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    args.output_dir = os.path.join(os.path.abspath(os.getcwd()), "outputs")
    args.func(args)


if __name__ == "__main__":
    main()

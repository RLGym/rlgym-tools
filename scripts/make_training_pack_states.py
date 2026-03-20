import argparse

from rlgym_tools.rocket_league.state_mutators.training_pack_mutator import TrainingPackMutator


def main(in_path: str, out_path: str):
    TrainingPackMutator.make_file(in_path, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate training pack states for all maps and modes.")
    parser.add_argument("--in-path", type=str, required=True,
                        help="Path to the training pack file(s)")
    parser.add_argument("--out-path", type=str, default="training_pack_states.jsonl.gz",
                        help="Path to save the generated states")
    args = parser.parse_args()
    main(
        in_path=args.in_path,
        out_path=args.out_path,
    )

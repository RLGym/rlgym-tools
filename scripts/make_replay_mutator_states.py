import argparse
import glob
import random
from concurrent.futures import ProcessPoolExecutor

from rlgym_tools.rocket_league.state_mutators.replay_mutator import ReplayMutator


def _make_file(mode: str, input_folder: str, output_folder: str) -> ReplayMutator:
    print(f"Processing {mode} replays...")
    replay_files = glob.glob(f"{input_folder}/{mode}/*.replay", recursive=True)
    print(f"Found {len(replay_files)} replay files.")
    replay_files = random.Random(0).sample(replay_files, k=10_000)
    ReplayMutator.make_file(
        replay_files=replay_files,
        output_path=f"{output_folder}/replay_frames_{mode}.npy",
        do_memory_map=True,
        max_num_players=int(mode[0]) * 2,
    )


def main(input_folder: str, output_folder: str):
    modes = ("1v1", "2v2", "3v3")
    with ProcessPoolExecutor(len(modes)) as ex:
        for _ in ex.map(_make_file, modes, [input_folder] * len(modes), [output_folder] * len(modes)):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate replay mutator states from replay files.")
    parser.add_argument("--input-folder", type=str, default="test_replays",
                        help="The folder containing the replay files.")
    parser.add_argument("--output-folder", type=str, default=".")
    args = parser.parse_args()
    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
    )

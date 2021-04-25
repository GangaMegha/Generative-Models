"""Run any of the other package capabilities using execution flags.

First argument has to specify, which script is to be executed.
Any other arguments will be passed to said script.
"""
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # Create standard collection of data sets.
    parser.add_argument('--num_obj', dest='num_obj', type=int, default=1)


    args, _ = parser.parse_known_args()
    script_args = sys.argv[2:]
    print(sys.argv)
    print(script_args)


    from envs import envs

    envs.main(script_args, args.num_obj)


    print("End of run_scripts.")

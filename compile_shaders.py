#!/usr/bin/env python3

import glob
import subprocess
import os
import argparse


def get_mtime(fpath) -> float:
    try:
        stat = os.stat(fpath)
        return stat.st_mtime
    except FileNotFoundError:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true', help='Remove compiled *.spv files')
    args = parser.parse_args()

    if args.clean:
        for file in glob.glob('resources/shaders/**', recursive=True):
            src_path = os.path.abspath(file)

            if src_path.endswith('.spv'):
                print('Remove {}'.format(file))
                os.unlink(src_path)
        return

    for file in glob.glob('resources/shaders/**', recursive=True):
        src_path = os.path.abspath(file)
        if not (src_path.endswith('.vert') or src_path.endswith('.frag')):
            continue

        dst_path = '{}.spv'.format(src_path)

        if get_mtime(dst_path) > get_mtime(src_path):
            continue

        print('Process {}'.format(file))
        ret = subprocess.run(['glslc', '-o', dst_path, src_path])
        ret.check_returncode()


if __name__ == '__main__':
    main()

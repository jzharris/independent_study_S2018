import shutil
from subprocess import call
import argparse

parser = argparse.ArgumentParser(description='Custom demo')
parser.add_argument('--file', default=None)

args = parser.parse_args()


def cmd_exists(cmd):
    return shutil.which(cmd) is not None

if not args.file:
    print('no obj file name given')
else:
    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    if cmd_exists('meshlab'):
        call(['meshlab', 'obj/{}.obj'.format(args.file)])
    else:
        print('Meshlab not found: please use visualization of your choice to view %s' %
              args.file)
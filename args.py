import argparse

parser = argparse.ArgumentParser(description="Use Reinforced DMP to adapt to new goals")
parser.add_argument('-ga', '--gain', type=float, default=20.0,
                    help="Set the gain of the DMP transformation system.")
parser.add_argument('-ng', '--num-gaussians', type=int, default=20,
                    help="Number of Gaussians")
parser.add_argument('-sb', '--stabilization', type=bool, default=False,
                    help="Add a stability term at end of trajectory")
parser.add_argument('-if', '--input-file', type=str, default='data/demo12.txt',
                    help="Input trajectory file")
parser.add_argument('-of', '--output-file', type=str, default='output.txt',
                    help="Output Gains file")
parser.add_argument('-p', '--show-plots', dest='show_plots', action='store_true',
                    help="Show plots at end of computation")
parser.add_argument('-np', '--no-plots', dest='show_plots', action='store_false',
                    help="Don't show plots at end of computation")
parser.add_argument('-i', '--use-inverse', dest='use_inverse', action='store_true',
                    help="Use inverse model to calculate velocities in RT")
parser.add_argument('-d', '--use-demo', dest='use_inverse', action='store_false',
                    help="Use demonstration file from --input instead of inverse model")
parser.add_argument('--use-cup', dest='objects', action='store_true',
                    help="Use orientation for cup")
parser.add_argument('--use-cube', dest='objects', action='store_false',
                    help="Use orientation for cube")
parser.add_argument('-w', '--window', type=int, default=5,
                    help="Window size for filtering")
parser.add_argument('-b', '--blends', type=int, default=10,
                    help="Number of blends for filtering")
parser.add_argument('-s', '--samples', type=int, default=10,
                    help="Number of paths for exploration")
parser.add_argument('-r', '--rate', type=float, default=0.5,
                    help="Number of possible paths to keep")
parser.set_defaults(show_plots=True, use_inverse=True, objects=True)
arg = parser.parse_args()

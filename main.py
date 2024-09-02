import os
import sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from game_runner import GameRunner

def main():
    game_runner = GameRunner()
    game_runner.run()

if __name__ == '__main__':
    main()
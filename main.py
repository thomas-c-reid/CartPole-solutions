import sys

sys.dont_write_bytecode = True        

from game_runner import GameRunner

def main():
    game_runner = GameRunner()
    game_runner.train()
    
    game_runner.visualize_trained_agent()
    
    

if __name__ == '__main__':
    main()
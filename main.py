from src.qlearning import QLearning1DGame
import os

visualize_mode = None
slow_mode = None

while True:
    print("Use visualize mode? (y/n) ")
    mode = input()
    if mode.lower() != "y" and mode.lower() != "n":
        print("Invalid input")
    else:
        if mode.lower() == "y":
            visualize_mode = True
        elif mode.lower() == "n":
            visualize_mode = False
            break

        while True:
            print("Use slow mode (for better observation)? (y/n)")
            mode_slow = input()
            if mode_slow.lower() == "y":
                slow_mode = True
                break
            elif mode_slow.lower() == "n":
                slow_mode = False
                break
            else:
                print("Invalid input")

        break

os.system("cls")
agent = QLearning1DGame()
agent.train(visualize_mode, slow_mode)
print("Q-table result")
print(agent.q_table)

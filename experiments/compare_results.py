print("\n[RESULT] Algorithm Benchmark Summary")
print("----------------------------------------")
print("Environment        : CartPole-v1 (reference task)")
print("Evaluation Protocol: Fixed episodes, ε=0 for DQN")
print("----------------------------------------")
print(f"DQN Mean Return    : {dqn_avg:.1f}")
print(f"PPO Mean Return    : {ppo_avg:.1f}")
print("----------------------------------------")

if dqn_avg > ppo_avg:
    print("[INTERPRETATION] Value-based methods are more sample-efficient for this task.")
else:
    print("[INTERPRETATION] Policy-gradient methods better captured task dynamics.")

print("[NOTE] Results are environment-dependent and not a general algorithm ranking.")

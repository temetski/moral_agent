from llm_fewShotPromptTraining import create_llm_env, few_shot_prompt_training, call_llm_with_state_action
from fromBeliefToRewardUsingDST import belief_to_reward
import numpy as np
import os
def main():
    
    credences = np.zeros((5, 5))
    # Set the diagonal elements
    for i in range(5):
        credences[i, i] = 1
    
      
    # print(os.environ['HOME'])
    api_key = os.environ.get("OPENAI_API_KEY")
    model = create_llm_env(api_key)
    final_prompt = few_shot_prompt_training()
    
    # This call goes inside RL Step loop
    belief_dict = call_llm_with_state_action("state","action",credences,model,final_prompt)
    print(belief_dict)
    reward_dict = belief_to_reward(belief_dict)
    
    for key, value in sorted(reward_dict.items()):
        print(f"{set(key)}: {value:.4f}")

if __name__ == "__main__":
    main()
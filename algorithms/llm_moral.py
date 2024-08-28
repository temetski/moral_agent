from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

import numpy as np
import re
import regex
import json
from collections import OrderedDict
import warnings
import os

moral_agent_types = ['consequentialist', 'deontologist', 'virtue', 'care', 'social justice']
moral_clusters = ['Consequentialist', 'Deontological', 'Virtue Ethics', 'Care Ethics', 'Social Justice Ethics']
_credences = np.eye(len(moral_agent_types), len(moral_clusters))

credences = {}
for i, ma in enumerate(moral_agent_types):
    agent_cred = {mc: _credences[i,j] for j, mc in enumerate(moral_clusters)}
    credences[ma] = agent_cred
credences['moral'] = None

def create_llm_env(key,model_name='llama3'):  
    if model_name=='llama3':
        base_url = 'http://10.249.72.3:8000/v1'
    else:
        base_url = None
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        api_key=key,  # if you prefer to pass api key in directly instaed of using env vars
        base_url=base_url,
        # organization="...",
        # other params...
    )
    return model
    
def few_shot_prompt_training():    
    example_all_moral_clusters = [
#         {
#         "question": """You are a train conductor facing a critical decision. There is a runaway trolley headed towards five workers on the main track. You can pull a lever to divert the trolley onto a side track, but there is one worker on that track. Additionally, you have the option to call for emergency assistance, though it might not arrive in time. Lastly, you could attempt to stop the trolley by placing an obstacle, but it could lead to unpredictable consequences. Should you pull the lever, call for help, place an obstacle, or do nothing and let the trolley continue on its path?

# A: Pull the lever.
# B: Call for emergency assistance.
# C: Place an obstacle.
# D: Do nothing.

# Behave as an agent that assigns the following credence values: {Consequentialist: 0.0, Deontological: 1.0, Virtue Ethics: 0.0, Care Ethics: 0.0, Social Justice Ethics: 0.0}. Using the credence confidence for moral theories, reason through what action you will take.""",
#         "answer":
#     """Let us think about this step by step. 
# What is the immediate benefit of pulling the lever?
# The immediate benefit is that four lives will be saved by sacrificing one. 
# What is the potential long-term consequence of this action?
# The potential long-term consequence includes the moral burden of directly causing one person's death. 
# What is the immediate benefit of calling for emergency assistance?
# The immediate benefit is the possibility of saving all six people without any loss of life. 
# What is the potential long-term consequence of this action?
# The potential long-term consequence includes the risk that the emergency assistance might not arrive in time, resulting in the loss of five lives. 
# What is the immediate benefit of placing an obstacle?
# The immediate benefit is that the trolley might be stopped without directly choosing to sacrifice any specific individual. 
# What is the potential long-term consequence of this action?
# The potential long-term consequence includes unpredictable outcomes, which could lead to even greater harm or no effect at all. 
# What is the immediate benefit of doing nothing?
# The immediate benefit is that the conductor does not directly intervene in the situation. 
# What is the potential long-term consequence of this action?
# The potential long-term consequence includes the certainty of five deaths and the moral burden of inaction.
# How does consequentialism weigh immediate benefits against long-term consequences?
# Consequentialism requires evaluating both immediate and long-term outcomes to determine which action maximizes overall well-being.
# Based these ethical credences, the probabilities of each action are: {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1 }"""
#         },
        {
        "question": """You are a train conductor facing a critical decision. There is a runaway trolley headed towards five workers on the main track. You can pull a lever to divert the trolley onto a side track, but there is one worker on that track. Additionally, you have the option to call for emergency assistance, though it might not arrive in time. Lastly, you could attempt to stop the trolley by placing an obstacle, but it could lead to unpredictable consequences. Should you pull the lever, call for help, place an obstacle, or do nothing and let the trolley continue on its path?

Action 0: Pull the lever.
Action 1: Call for emergency assistance.
Action 2: Place an obstacle.
Action 3: Do nothing.

Behave as an agent that assigns the following credence values: {Consequentialist: 0.0, Deontological: 1.0, Virtue Ethics: 0.0, Care Ethics: 0.0, Social Justice Ethics: 0.0}. Using the credence confidence for moral theories, reason through what action you will take.""",
        "answer":
    """Let us think about this step by step.

Based on the assigned credence values, Deontological ethics is the most important moral cluster for my considerations. Deontological ethics emphasizes the importance of adhering to moral rules and duties. Since it prioritizes adherence to moral duties and rules over the consequences of actions, we should focus on the inherent rightness or wrongness of the actions themselves. Let us assess all of the actions based on this lens:


Pull the lever: Pulling the lever would directly violate the duty to not intentionally harm an innocent person.
Call for emergency assistance: Calling for emergency assistance aligns with the duty to seek help and preserve life without directly causing harm to others. However, given the urgency of the situation, this might result in inaction, and potentially fail at saving the workers.
Place an obstacle: Placing an obstacle involves taking an action that could have unpredictable outcomes, which might violate the duty to avoid causing unintended harm.
Do nothing: Doing nothing may align with the principle of non-interference, respecting the natural course of events without directly causing harm.

Following deontological ethics, the most consistent course of action would be to not actively causing harm to anyone. This would adhere to my duty to avoid taking a direct action that leads to someone's death, even though this results in allowing harm to happen due to external forces beyond my control.

Based these ethical credences, the probabilities of each action are: {"0": 0.05, "1": 0.3, "2": 0.05, "3": 0.6 }"""
        },
#         {
#         "question": """A company is deciding whether to implement a costly environmental protection measure. This measure will significantly reduce pollution and improve public health but will also lead to increased product prices and potential job losses. Additionally, the company could partially implement the measure to balance costs and benefits, or invest in community health initiatives as an alternative solution. Should the company fully implement the measure, partially implement it, invest in community health, or not implement the measure at all?

# 0: Fully implement the health measures.
# 1: Partially implement the health measures.
# 2: Invest in community health initiatives.
# 3: Do not implement the health measures.

# Behave as an agent that assigns the following credence values: {Consequentialist: 0.0, Deontological: 0.0, Virtue Ethics: 1.0, Care Ethics: 0.0, Social Justice Ethics: 0.0}. Using the credence confidence for moral theories, reason through what action you will take.""",
#         "answer":
#     """Let us think about this step by step.

# Based on the credence values given, my priorities are in line with virtue ethics. In deciding to implement environmental protection measures, the virtues of responsibility, compassion, and justice are relevant
#     What virtues are relevant to the decision of implementing the environmental protection measure?
# Relevant virtues include responsibility, compassion, and justice.
# How does the virtue of responsibility apply to this decision?
# The virtue of responsibility involves acknowledging the company's duty to protect the environment and public health, acting as a steward for future generations.
# How does the virtue of compassion apply to this decision?
# The virtue of compassion emphasizes the importance of caring for the well-being of the community and employees, ensuring that actions taken do not cause unnecessary harm.
# How does the virtue of justice apply to this decision?
# The virtue of justice involves fairness in considering the impacts on all stakeholders, balancing the needs of the community, employees, and consumers.
# How does virtue ethics evaluate the character and motivations of the decision-makers?
# Virtue ethics evaluates the decision based on the moral character and intentions of the decision-makers, aiming for actions that reflect virtuous traits and contribute to a flourishing society.

# Based these ethical credences, the probabilities of each action are: {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1 }"""
#         },
        {
        "question": """A city is deciding whether to allocate a significant portion of its budget to building a new public park in an underserved neighborhood. This park will provide recreational opportunities and improve the quality of life for residents but will also mean reducing funds for other services such as public transportation, healthcare, and education. Additionally, the city could opt to build smaller parks in multiple neighborhoods or invest in improving existing facilities. Should the city allocate the budget to build the new park, distribute the funds to smaller projects, improve existing facilities, or maintain the current budget distribution?

Action A: Build the new park.
Action B: Distribute funds to smaller projects.
Action C: Improve existing facilities.
Action D: Maintain the current budget distribution.

Behave as an agent that assigns the following credence values: {Consequentialist: 0.0, Deontological: 0.0, Virtue Ethics: 0.0, Care Ethics: 0.0, Social Justice Ethics: 1.0}. Using the credence confidence for moral theories, reason through what action you will take.""",
        "answer":
    """Let us think about this step by step.

Based on the assigned credence values, Social Justice is the most important moral cluster for my considerations. Social justice evaluates the distribution of resources by ensuring that benefits and burdens are shared fairly, prioritizing the needs of the most disadvantaged and promoting equity. Let us evaluate all of the actions based on this lens:

Build the new park: Building a new park in the underserved neighborhood can provide recreational opportunities and improve the quality of life for residents in the underserved area, which promotes equity and inclusion. However, this can also reduce funds for other critical services such as public transportation, healthcare, and education, which may disproportionately affect other vulnerable populations.
Distribute funds to smaller projects: Doing so can provide widespread benefits, improving multiple areas and promoting a sense of fairness and inclusion across the city. Its potential drawbacks include a diluted impact, where smaller projects may not significantly improve any single area as much as a large, focused investment would.
Improving existing facilities: By improving existing facilities, we can address immediate needs and enhance the quality of life for current users, promoting efficient use of resources and incremental improvements. 
Maintain the current budget distribution: While this avoids disruption and ensures continued support for existing services, it may also perpetuate existing inequalities and fail to address the needs of underserved areas.

Given the full weight of Social Justice Ethics, the city should prioritize rectifying historical inequities. Building the new park (Action A) would be the most direct action to address the specific needs of the underserved neighborhood, fostering greater equality and community well-being.

Based these ethical credences, the probabilities of each action are: {"A": 0.6, "B": 0.3, "C": 0.1, "D": 0.0 }"""
        },
        {
        "question": """A city is deciding whether to allocate a significant portion of its budget to building a new public park in an underserved neighborhood. This park will provide recreational opportunities and improve the quality of life for residents but will also mean reducing funds for other services such as public transportation, healthcare, and education. Additionally, the city could opt to build smaller parks in multiple neighborhoods or invest in improving existing facilities. Should the city allocate the budget to build the new park, distribute the funds to smaller projects, improve existing facilities, or maintain the current budget distribution?

Action build: Build the new park.
Action distribute: Distribute funds to smaller projects.
Action improve: Improve existing facilities.
Action maintain: Maintain the current budget distribution.

Behave as an agent that assigns the following credence values: {Consequentialist: 0.0, Deontological: 0.0, Virtue Ethics: 0.0, Care Ethics: 1.0, Social Justice Ethics: 0.0}. Using the credence confidence for moral theories, reason through what action you will take.""",
        "answer":
    """Let us think about this step by step.

Based on the assigned credence values, Care Ethics is the most important moral cluster for my considerations. Care ethics evaluates the distribution of resources by emphasizing relationships, interconnectedness, and the well-being of all community members, prioritizing actions that nurture and sustain care. Let us evaluate all of the actions based on this lens:

build: Building the new park in the underserved neighborhood can provide recreational opportunities and improving the quality of life for residents in an underserved area, fostering a sense of community and care. This could come at the cost of reducing funds for other critical services such as public transportation, healthcare, and education, which may negatively affect the well-being of other community members.
distribute: Distributing funds to smaller projects can spread benefits more widely, showing care for multiple communities and fostering broader well-being. 
On this other hand, this approach includes the possibility that smaller projects may not significantly impact any single area, potentially leading to less effective outcomes in terms of care and community building.
improve: Improving existing facilities can address the needs of current users, demonstrating ongoing care and commitment to maintaining and enhancing community resources. 

maintain: Maintaining the current budget distribution ensures continuity of existing services, but it may overlook opportunities to address unmet needs and enhance care in underserved areas.

Under Care Ethics, the focus is on providing care to those who need it most, ensuring that no group is neglected, and fostering relationships across communities. Therefore, the best course of action would likely be Action B: Distribute funds to smaller projects. This approach spreads the benefits across multiple neighborhoods, ensuring that the city shows care for multiple groups of people without severely reducing critical services such as transportation, healthcare, and education. Additionally, this strategy is more likely to foster positive relationships across different communities by showing that the city is attentive to the needs of all its residents, not just one underserved neighborhood.

Based these ethical credences, the probabilities of each action are: {"build": 0.1, "distribute": 0.6, "improve": 0.3, "maintain": 0.0}"""
        }
    ]


    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{question}"),
            ("assistant", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=example_all_moral_clusters,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a moral agent that is capable of following the following  moral clusters.

The Consequentialist Ethics moral cluster provides a strong focus on Focus on outcomes and results of actions. The key principle that drives this moral code is Maximizing overall good/well-being.
This moral cluster is further guided by the following ethical theories:
- Classical Utilitarianism: This ethical theory adheres to the following key concepts: Greatest good for the greatest number, Hedonic calculus. When making decisions, this theory must take into account the following factors: Pleasure, Pain, Aggregate welfare.
- Preference Utilitarianism: This ethical theory adheres to the following key concepts: Satisfaction of preferences, Informed desires. When making decisions, this theory must take into account the following factors: Individual preferences, Long-term satisfaction.
- Rule Utilitarianism: This ethical theory adheres to the following key concepts: Rules that maximize utility, Indirect consequentialism. When making decisions, this theory must take into account the following factors: Rule adherence, Overall societal benefit.
- Ethical Egoism: This ethical theory adheres to the following key concepts: Self-interest, Rational selfishness. When making decisions, this theory must take into account the following factors: Personal benefit, Long-term self-interest.
- Prioritarianism: This ethical theory adheres to the following key concepts: Prioritizing the worse-off, Weighted benefit. When making decisions, this theory must take into account the following factors: Inequality, Marginal utility, Relative improvement.

The Deontological Ethics moral cluster provides a strong focus on Focus on adherence to moral rules and obligations. The key principle that drives this moral code is Acting according to universal moral laws.
This moral cluster is further guided by the following ethical theories:
- Kantian Ethics: This ethical theory adheres to the following key concepts: Categorical Imperative, Universalizability, Treating humans as ends. When making decisions, this theory must take into account the following factors: Universality, Respect for autonomy, Moral duty.
- Prima Facie Duties: This ethical theory adheres to the following key concepts: Multiple duties, Situational priority. When making decisions, this theory must take into account the following factors: Fidelity, Reparation, Gratitude, Justice, Beneficence.
- Rights Based Ethics: This ethical theory adheres to the following key concepts: Individual rights, Non-interference. When making decisions, this theory must take into account the following factors: Liberty, Property rights, Human rights.
- Divine Command Theory: This ethical theory adheres to the following key concepts: God's will as moral standard, Religious ethics. When making decisions, this theory must take into account the following factors: Religious teachings, Divine revelation, Scriptural interpretation.

The Virtue Ethics moral cluster provides a strong focus on Focus on moral character and virtues of the agent. The key principle that drives this moral code is Cultivating virtuous traits and dispositions.
This moral cluster is further guided by the following ethical theories:
- Aristotelian Virtue Ethics: This ethical theory adheres to the following key concepts: Golden mean, Eudaimonia, Practical wisdom. When making decisions, this theory must take into account the following factors: Courage, Temperance, Justice, Prudence.
- Neo Aristotelian Virtue Ethics: This ethical theory adheres to the following key concepts: Modern virtue interpretation, Character development. When making decisions, this theory must take into account the following factors: Integrity, Honesty, Compassion, Resilience.
- Confucian Ethics: This ethical theory adheres to the following key concepts: Ren (benevolence), Li (propriety), Harmonious society. When making decisions, this theory must take into account the following factors: Filial piety, Social harmony, Self-cultivation.
- Buddhist Ethics: This ethical theory adheres to the following key concepts: Four Noble Truths, Eightfold Path, Karma. When making decisions, this theory must take into account the following factors: Compassion, Non-attachment, Mindfulness.

The Care Ethics moral cluster provides a strong focus on Focus on relationships, care, and context. The key principle that drives this moral code is Maintaining and nurturing relationships.
This moral cluster is further guided by the following ethical theories:
- Noddings Care Ethics: This ethical theory adheres to the following key concepts: Empathy, Responsiveness, Attentiveness. When making decisions, this theory must take into account the following factors: Relationships, Context, Emotional intelligence.
- Moral Particularism: This ethical theory adheres to the following key concepts: Situational judgment, Anti-theory. When making decisions, this theory must take into account the following factors: Contextual details, Moral perception.
- Ubuntu Ethics: This ethical theory adheres to the following key concepts: Interconnectedness, Community, Humanness through others. When making decisions, this theory must take into account the following factors: Collective welfare, Shared humanity, Reciprocity.
- Feminist Ethics: This ethical theory adheres to the following key concepts: Gender perspective, Power dynamics, Inclusivity. When making decisions, this theory must take into account the following factors: Gender equality, Marginalized voices, Intersectionality.

The Social Justice Ethics moral cluster provides a strong focus on Focus on fairness, equality, and social contracts. The key principle that drives this moral code is Creating just societal structures.
This moral cluster is further guided by the following ethical theories:
- Rawlsian Justice: This ethical theory adheres to the following key concepts: Veil of ignorance, Difference principle. When making decisions, this theory must take into account the following factors: Fairness, Equal opportunity, Social inequality.
- Contractarianism: This ethical theory adheres to the following key concepts: Social contract, Mutual advantage. When making decisions, this theory must take into account the following factors: Rational self-interest, Cooperation, Agreement.
- Capabilities Approach: This ethical theory adheres to the following key concepts: Human capabilities, Freedom to achieve well-being. When making decisions, this theory must take into account the following factors: Individual capabilities, Social opportunities, Personal choice.
- Environmental Ethics: This ethical theory adheres to the following key concepts: Sustainability, Biocentrism, Intergenerational justice. When making decisions, this theory must take into account the following factors: Environmental impact, Future generations, Ecological balance.

You will make your decisions on complex scenarios following the above moral codes. You should explain your reasoning on how you choose your actions. At the end and last line of you answer, state the action and probability of belief for that choice as a json with the action as keys and probability as values. The sum of all beliefs should be equal to 1."""),
            few_shot_prompt,
            ("user", "{scenario}"),
        ])
    return final_prompt

def call_llm_with_state_action(scenario_prompt,actionsets,state,action,credences,model,final_prompt):        
    question_text_all = []
    question_response_dict = {}
    i=0
    votes = []
    belief_dict = {ma: 0 for ma in credences.keys()}
    for moral_agent, credence_set in credences.items():
        if credence_set is not None:
            weighted_Credences = ', '.join(f'{k}: {v}' for k, v in credence_set.items())
            credence_text = f"Behave as an agent that assigns the following credence values: {{{weighted_Credences}}}. Using the credence confidence for moral theories, reason through what action you will take."
        else:
            credence_text = f"Behave as moral agent. Using moral theories, reason through what action you will take."
        question_text = f"{scenario_prompt} {state}\n\n{action}\n\n{credence_text}"
        question_text_all.append(question_text)
        # question_text_log.append(question_text)
        # print(question_text)
        formattedChatPrompt = final_prompt.format_messages(scenario=question_text)
        response = model.invoke(formattedChatPrompt)
        prompt_token_usage = response.response_metadata['token_usage']['prompt_tokens']
        completion_token_usage = response.response_metadata['token_usage']['completion_tokens']
        total_token_usage = response.response_metadata['token_usage']['total_tokens']

        # response.usage.total_tokens()
        # response_text_log.append(response.content)
        # print(response.content)
        question_response_dict[question_text] = response.content
        dict_pattern = r'\{(?:[^{}]|(?R))*\}'
        pattern = regex.compile(dict_pattern)

        if response.response_metadata['finish_reason']!='stop':
            print("LLM ran out of token. Missing response content. Possible infinite loop")
        # Search for the pattern in the text
        # Use findall to get all matches
        matches = pattern.findall(response.content)
        # Check if there are any matches
        if matches:
            # Access the first match if it exists
            match = matches[0]
        else:
            # Handle the case where no matches are found
            print("No matches found.")
            match = '{"0": 0.0, "1": 0.0, "2": 0.0}'
    

        beliefs = json.loads(match, object_pairs_hook=OrderedDict) # ensure dictionary does not reorder choices
        # print(beliefs)
        # Fill missing values if LLM does not output beliefs for other actions
        for actionset in actionsets:
            assert len(actionset)==1
            item = list(actionset)[0]
            if list(actionset)[0] not in beliefs.keys():
                beliefs[item] = 0

        # remove erroneous actions added by the LLM
        keys_to_del = []
        for key in beliefs.keys():
            if frozenset([key]) not in actionsets:
                keys_to_del.append(key)
        for key in keys_to_del:
            del beliefs[key] 
        assert len(actionsets) == len(beliefs) 

        one_row = np.array([beliefs[list(actionset)[0]] for actionset in actionsets], dtype=float)
        sum_of_belief = one_row.sum()
        if np.all(one_row==0):
            warnings.warn("Beliefs cannot be zero. Setting beliefs to be equal")
            one_row += 1./len(one_row)
        if not np.isclose(sum_of_belief, 1):
            warnings.warn("The sum of beliefs outputted by LLM is not equal to 1. Proceeding to normalize values")
            one_row /= one_row.sum()
        # assert sum_of_belief>0, f"ERROR: beliefs cannot be zero. \n{response.content}"
        belief_dict[moral_agent] = one_row
    return belief_dict, question_response_dict,total_token_usage



if __name__=="__main__":
    import gymnasium as gym
    import os
    import sys
    from dempster_shafer import belief_to_reward
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))

    # debugging examples
    model_name = "llama3"
    # model_name = "gpt-4o-mini"

    # state = "You are the captain of a lifeboat that has a maximum capacity of 10 people, including yourself. Currently, there are 10 people on board, and the lifeboat is in a stable condition. However, you encounter another group of 5 people stranded in the water who will certainly drown. if not rescued. If you take all 5 additional people on board, the lifeboat will become overcrowded and there is a high risk it will capsize, potentially resulting in the death of everyone on board."
    # action = "Action A: Take all 5 additional people on board. \nAction B: Take as many people as you can safely accommodate, and leave the rest. \nAction C: Rotate rescuing people by keeping the boat at its capacity while trying to transfer some to nearby boats or signaling for help. \nAction D: Do not take any additional people on board and prioritize the safety of those already on the lifeboat."
    api_key_coss = os.environ.get("OPENAI_API_KEY", "none")
    model = create_llm_env(api_key_coss,model_name)

    # scenario_prompt = "You are a driving agent to simulate autonomous driving. Cars could be driving in all of the lanes and sometimes there are old grandma with dementia trapped in the traffic on some lanes. Your primary objective is to avoid collisions. You will collide based on the cars on your current lane and the lane you decide to take. Similarly, you will be able to rescue the grandma on your current lane and the lane you decide to take. You will collide with a car if you come within 1 unit distance and can rescue up grandma if you are within 3 unit distance."
    
    # state = "You are currently on lane 0. The current lane has car at 7 unit distance and has no grandma. The lane on the right has car at 5 unit distance and has no grandma. The lane on the left does not exist and you cannot take it. You can perform the following actions:"

#     action = """Action A: Going up brings you closer to 0 crying and 0 sleeping babies
# Action B: Going down brings you closer to 5 crying and 0 sleeping babies.
# Action C: Going left brings you closer to 1 crying and 1 sleeping babies.
# Action D: Going right brings you closer to 0 crying and 2 sleeping babies."""

    env = gym.make('environments.milk:FindMilk-v3', render_mode='ansi', validate=True)
    env.reset()
    new_pos = (6,9)
    # new_pos = (4,2)
    # env.unwrapped.milk_pos = (4,6)
    env.generate_state(new_pos)
    actionsets = [frozenset([str(k)]) for k in env.action_mapper.keys()]

    scenario_prompt = env.get_scenario_prompt()
    state, action = env.state_as_text()

    # credences = credences[[0,1],:]
    final_prompt = few_shot_prompt_training()
    beliefs, question_response_dict, _ = call_llm_with_state_action(scenario_prompt,actionsets,state,action,credences,model,final_prompt)
    print(env.render())
    print(list(question_response_dict.keys())[0])
    for q, r in question_response_dict.items():
        print(r)
    reward_dict = belief_to_reward(beliefs, actionsets)
    print(reward_dict)

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

import numpy as np
import re
import regex
import json

def create_llm_env(key):  
    model = ChatOpenAI(
        model="llama3",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=key,  # if you prefer to pass api key in directly instaed of using env vars
        base_url='http://10.249.72.3:8000/v1',
        # organization="...",
        # other params...
    )
    return model
    
def few_shot_prompt_training():    
    example_all_moral_clusters = [
        {
        "question": """You are a train conductor facing a critical decision. There is a runaway trolley headed towards five workers on the main track. You can pull a lever to divert the trolley onto a side track, but there is one worker on that track. Additionally, you have the option to call for emergency assistance, though it might not arrive in time. Lastly, you could attempt to stop the trolley by placing an obstacle, but it could lead to unpredictable consequences. Should you pull the lever, call for help, place an obstacle, or do nothing and let the trolley continue on its path?

Action A: Pull the lever.
Action B: Call for emergency assistance.
Action C: Place an obstacle.
Action D: Do nothing.

Behave as an agent that assigns the following credence values: {{Consequentialist: 1.0, Deontological: 0.0, Virtue Ethics: 0.0, Care Ethics: 0.0, Social Justice Ethics: 0.0}}. Reason through using the credence confidence for moral theories.""",
        "answer":
    """Let us think about this step by step. 
What is the immediate benefit of pulling the lever?
The immediate benefit is that four lives will be saved by sacrificing one. 
What is the potential long-term consequence of this action?
The potential long-term consequence includes the moral burden of directly causing one person's death. 
What is the immediate benefit of calling for emergency assistance?
The immediate benefit is the possibility of saving all six people without any loss of life. 
What is the potential long-term consequence of this action?
The potential long-term consequence includes the risk that the emergency assistance might not arrive in time, resulting in the loss of five lives. 
What is the immediate benefit of placing an obstacle?
The immediate benefit is that the trolley might be stopped without directly choosing to sacrifice any specific individual. 
What is the potential long-term consequence of this action?
The potential long-term consequence includes unpredictable outcomes, which could lead to even greater harm or no effect at all. 
What is the immediate benefit of doing nothing?
The immediate benefit is that the conductor does not directly intervene in the situation. 
What is the potential long-term consequence of this action?
The potential long-term consequence includes the certainty of five deaths and the moral burden of inaction.
How does consequentialism weigh immediate benefits against long-term consequences?
Consequentialism requires evaluating both immediate and long-term outcomes to determine which action maximizes overall well-being.
Based these ethical credences, the probabilities of each action are: {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1 }"""
        },
        {
        "question": """You are a train conductor facing a critical decision. There is a runaway trolley headed towards five workers on the main track. You can pull a lever to divert the trolley onto a side track, but there is one worker on that track. Additionally, you have the option to call for emergency assistance, though it might not arrive in time. Lastly, you could attempt to stop the trolley by placing an obstacle, but it could lead to unpredictable consequences. Should you pull the lever, call for help, place an obstacle, or do nothing and let the trolley continue on its path?

Action A: Pull the lever.
Action B: Call for emergency assistance.
Action C: Place an obstacle.
Action D: Do nothing.

Behave as an agent that assigns the following credence values: {{Consequentialist: 0.0, Deontological: 1.0, Virtue Ethics: 0.0, Care Ethics: 0.0, Social Justice Ethics: 0.0}}. Reason through using the credence confidence for moral theories.""",
        "answer":
    """Let us think about this step by step. 
What is the deontological principle behind pulling the lever?
Deontological ethics emphasizes the importance of adhering to moral rules and duties. Pulling the lever would directly violate the duty to not intentionally harm an innocent person. 
What is the deontological principle behind calling for emergency assistance?
Calling for emergency assistance aligns with the duty to seek help and preserve life without directly causing harm to others. 
What is the deontological principle behind placing an obstacle?
Placing an obstacle involves taking an action that could have unpredictable outcomes, which might violate the duty to avoid causing unintended harm. 
What is the deontological principle behind doing nothing?
Doing nothing may align with the principle of non-interference, respecting the natural course of events without directly causing harm.
How does deontological ethics weigh the importance of adhering to moral duties against the outcomes of actions?
Deontological ethics prioritizes adherence to moral duties and rules over the consequences of actions, focusing on the inherent rightness or wrongness of the actions themselves.
Based these ethical credences, the probabilities of each action are: {"A": 0.1, "B": 0.5, "C": 0.2, "D": 0.2 }"""
        },
        {
        "question": """A company is deciding whether to implement a costly environmental protection measure. This measure will significantly reduce pollution and improve public health but will also lead to increased product prices and potential job losses. Additionally, the company could partially implement the measure to balance costs and benefits, or invest in community health initiatives as an alternative solution. Should the company fully implement the measure, partially implement it, invest in community health, or not implement the measure at all?

Action 0: Fully implement the health measures.
Action 1: Partially implement the health measures.
Action 2: Invest in community health initiatives.
Action 3: Do not implement the health measures.

Behave as an agent that assigns the following credence values: {{Consequentialist: 0.0, Deontological: 0.0, Virtue Ethics: 1.0, Care Ethics: 0.0, Social Justice Ethics: 0.0}}. Reason through using the credence confidence for moral theories.""",
        "answer":
    """Let us think about this step by step. 
What virtues are relevant to the decision of implementing the environmental protection measure?
Relevant virtues include responsibility, compassion, and justice.
How does the virtue of responsibility apply to this decision?
The virtue of responsibility involves acknowledging the company's duty to protect the environment and public health, acting as a steward for future generations.
How does the virtue of compassion apply to this decision?
The virtue of compassion emphasizes the importance of caring for the well-being of the community and employees, ensuring that actions taken do not cause unnecessary harm.
How does the virtue of justice apply to this decision?
The virtue of justice involves fairness in considering the impacts on all stakeholders, balancing the needs of the community, employees, and consumers.
How does virtue ethics evaluate the character and motivations of the decision-makers?
Virtue ethics evaluates the decision based on the moral character and intentions of the decision-makers, aiming for actions that reflect virtuous traits and contribute to a flourishing society.
Based these ethical credences, the probabilities of each action are: {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1 }"""
        },
        {
        "question": """A city is deciding whether to allocate a significant portion of its budget to building a new public park in an underserved neighborhood. This park will provide recreational opportunities and improve the quality of life for residents but will also mean reducing funds for other services such as public transportation, healthcare, and education. Additionally, the city could opt to build smaller parks in multiple neighborhoods or invest in improving existing facilities. Should the city allocate the budget to build the new park, distribute the funds to smaller projects, improve existing facilities, or maintain the current budget distribution?

Action A: Build the new park.
Action B: Distribute funds to smaller projects.
Action C: Improve existing facilities.
Action D: Maintain the current budget distribution.

Behave as an agent that assigns the following credence values: {{Consequentialist: 0.0, Deontological: 0.0, Virtue Ethics: 0.0, Care Ethics: 0.0, Social Justice Ethics: 1.0}}. Reason through using the credence confidence for moral theories.""",
        "answer":
    """Let us think about this step by step. 
What are the positive outcomes of building the new park in the underserved neighborhood?
The positive outcomes include providing recreational opportunities and improving the quality of life for residents in an underserved area, promoting equity and inclusion. 
What are the negative outcomes of allocating the budget to the new park?
The negative outcomes include reducing funds for other critical services such as public transportation, healthcare, and education, which may disproportionately affect other vulnerable populations.
How does social justice evaluate the distribution of resources?
Social justice evaluates the distribution of resources by ensuring that benefits and burdens are shared fairly, prioritizing the needs of the most disadvantaged and promoting equity. 
What are the potential benefits of distributing funds to smaller projects in multiple neighborhoods?
Distributing funds to smaller projects can provide widespread benefits, improving multiple areas and promoting a sense of fairness and inclusion across the city. 
What are the potential drawbacks of this approach?
The potential drawbacks include a diluted impact, where smaller projects may not significantly improve any single area as much as a large, focused investment would.
How does improving existing facilities compare to building new ones in terms of social justice?
Improving existing facilities can address immediate needs and enhance the quality of life for current users, promoting efficient use of resources and incremental improvements. 
What are the potential benefits and drawbacks of maintaining the current budget distribution?
Maintaining the current budget distribution avoids disruption and ensures continued support for existing services, but it may also perpetuate existing inequalities and fail to address the needs of underserved areas.
Based these ethical credences, the probabilities of each action are: {"A": 0.3, "B": 0.4, "C": 0.2, "D": 0.1 }"""
        },
        {
        "question": """A city is deciding whether to allocate a significant portion of its budget to building a new public park in an underserved neighborhood. This park will provide recreational opportunities and improve the quality of life for residents but will also mean reducing funds for other services such as public transportation, healthcare, and education. Additionally, the city could opt to build smaller parks in multiple neighborhoods or invest in improving existing facilities. Should the city allocate the budget to build the new park, distribute the funds to smaller projects, improve existing facilities, or maintain the current budget distribution?

Action build: Build the new park.
Action distribute: Distribute funds to smaller projects.
Action improve: Improve existing facilities.
Action maintain: Maintain the current budget distribution.

Behave as an agent that assigns the following credence values: {{Consequentialist: 0.0, Deontological: 0.0, Virtue Ethics: 0.0, Care Ethics: 1.0, Social Justice Ethics: 0.0}}. Reason through using the credence confidence for moral theories.""",
        "answer":
    """Let us think about this step by step. 
What are the positive outcomes of building the new park in the underserved neighborhood?
The positive outcomes include providing recreational opportunities and improving the quality of life for residents in an underserved area, fostering a sense of community and care. 
What are the negative outcomes of allocating the budget to the new park?
The negative outcomes include reducing funds for other critical services such as public transportation, healthcare, and education, which may negatively affect the well-being of other community members.
How does care ethics evaluate the distribution of resources?
Care ethics evaluates the distribution of resources by emphasizing relationships, interconnectedness, and the well-being of all community members, prioritizing actions that nurture and sustain care. 
What are the potential benefits of distributing funds to smaller projects in multiple neighborhoods?
Distributing funds to smaller projects can spread benefits more widely, showing care for multiple communities and fostering broader well-being. 
What are the potential drawbacks of this approach?
The potential drawbacks include the possibility that smaller projects may not significantly impact any single area, potentially leading to less effective outcomes in terms of care and community building.
How does improving existing facilities compare to building new ones in terms of care ethics?
Improving existing facilities can address the needs of current users, demonstrating ongoing care and commitment to maintaining and enhancing community resources. 
What are the potential benefits and drawbacks of maintaining the current budget distribution?
Maintaining the current budget distribution ensures continuity of existing services, but it may overlook opportunities to address unmet needs and enhance care in underserved areas.
Based these ethical credences, the probabilities of each action are: {"build": 0.4, "distribute": 0.3, "improve": 0.2, "maintain": 0.1}"""
        }
    ]


    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["question"],
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

You will make your decisions on complex scenarios following the above moral codes. You should explain your reasoning on how you choose your actions. At the end and last line of you answer, state the action and probability of belief for that choice as a dictionary of key:value pairs. Do not write reasoning or anything else after the lastline. The sum of all beliefs should be equal to 1."""),
            few_shot_prompt,
            ("human","{question}"),
            # ("human", "You are the captain of a lifeboat that has a maximum capacity of 10 people, including yourself. Currently, there are 10 people on board, and the lifeboat is in a stable condition.
However, you encounter another group of 5 people stranded in the water who will certainly drown if not rescued. If you take all 5 additional people on board, the lifeboat will become overcrowded and there is a high risk it will capsize, potentially resulting in the death of everyone on board. \nAction A: I will rescue 5 additional people. \nAction B: I will not rescue 5 additional people.\n{question}"),
            
        ])
    return final_prompt

def call_llm_with_state_action(scenario_prompt,state,action,credences,model,final_prompt):        
    question_text_all = []
    i=0
    votes = []
    belief_dict = {i: 0 for i in range(5)}
    for c in credences:
        # c_list = c.split(" ")
        c_Consequentialist = credences[i][0];c_Deontological = credences[i][1];c_Virtue = credences[i][2];c_Care = credences[i][3];c_SocialJustice = credences[i][4]
   
        weighted_Credences = f"Consequentialist: {c_Consequentialist}, Deontological: {c_Deontological}, Virtue Ethics: {c_Virtue}, Care Ethics: {c_Care}, Social Justice Ethics: {c_SocialJustice}"
        # print(weighted_Credences)

        # state = "You are the captain of a lifeboat that has a maximum capacity of 10 people, including yourself. Currently, there are 10 people on board, and the lifeboat is in a stable condition.
However, you encounter another group of 5 people stranded in the water who will certainly drown. if not rescued. If you take all 5 additional people on board, the lifeboat will become overcrowded and there is a high risk it will capsize, potentially resulting in the death of everyone on board."
        # action = "Action A: Take all 5 additional people on board. \nAction B: Take as many people as you can safely accommodate, and leave the rest. \nAction C: Rotate rescuing people by keeping the boat at its capacity while trying to transfer some to nearby boats or signaling for help. \nAction D: Do not take any additional people on board and prioritize the safety of those already on the lifeboat."

        credence_text = "Behave as an agent that assigns the following credence values: {{" + weighted_Credences + "}}. Reason through using the credence confidence for moral theories."
        question_text = f"{scenario_prompt}{state}\n\n{action}\n\n{credence_text}"
        question_text_all.append(question_text)
        # print(question_text)
        # formattedChatPrompt = final_prompt.format_messages(question="You are the captain of a lifeboat that has a maximum capacity of 10 people, including yourself. Currently, there are 10 people on board, and the lifeboat is in a stable condition.
However, you encounter another group of 5 people stranded in the water who will certainly drown if not rescued. If you take all 5 additional people on board, the lifeboat will become overcrowded and there is a high risk it will capsize, potentially resulting in the death of everyone on board. \n Action A: I will rescue 5 additional people. \n Action B: I will not rescue 5 additional people. Behave as an agent that assigns the following credence values: {}. Reason through using the credence as weighted confidence for moral theories and output 
What action will you choose as Final answer:?
")
        formattedChatPrompt = final_prompt.format_messages(question=question_text)
        response = model.invoke(formattedChatPrompt)
        # print("Response Content: ", response.content)
        value = response.content.split('\n')

        print(response.content)

        dict_pattern = r'\{(?
:[^{}]|(?
R))*\}'
        pattern = regex.compile(dict_pattern)

        
        # Search for the pattern in the text
        match = pattern.findall(response.content)[0]

        beliefs = json.loads(match)
        one_row = '_'.join([str(x) for x in beliefs.values()])
        sum_of_belief = sum(beliefs.values())
        np.testing.assert_almost_equal(sum_of_belief, 1, err_msg="The sum of beliefs outputted by LLM is not equal to 1")
        # print(one_row)
        belief_dict[i] = one_row
        i+=1
    return belief_dict

if __name__=="__main__":
    # debugging examples
    NUM_MORAL = 5
    credences = np.zeros((5, NUM_MORAL))
    # Set the diagonal elements
    for i in range(NUM_MORAL):
        credences[i, i] = 1

    model = create_llm_env("no-key")

    scenario_prompt = "You are tasked with finding milk as quickly as possible in a room filled with babies. Most babies are sleeping, however, there are some babies that are crying. Passing by the parts of the room with a baby will disturb the baby, but passing by a crying baby will pacify them. Currently there are 9 crying babies in the room."
    
    state = "You are currently at position (9, 7) and the milk is at position (9,9). You can perform the following actions."

    action = """Action A: Going up brings you closer to 0 crying and 0 sleeping babies
Action B: Going down brings you closer to 10 crying and 0 sleeping babies.
Action C: Going left brings you closer to 1 crying and 1 sleeping babies.
Action D: going right brings you closer to 0 crying and 2 sleeping babies."""

    final_prompt = few_shot_prompt_training()
    call_llm_with_state_action(scenario_prompt,state,action,credences,model,final_prompt)

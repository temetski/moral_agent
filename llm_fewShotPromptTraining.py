from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

import numpy as np
import re


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
        "question": "You are a train conductor facing a critical decision. There is a runaway trolley headed towards five workers on the main track. You can pull a lever to divert the trolley onto a side track, but there is one worker on that track. Additionally, you have the option to call for emergency assistance, though it might not arrive in time. Lastly, you could attempt to stop the trolley by placing an obstacle, but it could lead to unpredictable consequences. Should you pull the lever, call for help, place an obstacle, or do nothing and let the trolley continue on its path? Action A: Pull the lever. Action B: Call for emergency assistance. Action C: Place an obstacle. Action D: Do nothing. Explain your reasoning based on consequentialist principles.",
        "answer": 
    """
    Are follow-up questions needed here: Yes.
    Follow up: What is the immediate benefit of pulling the lever?
    Intermediate answer: The immediate benefit is that four lives will be saved by sacrificing one.
    Follow up: What is the potential long-term consequence of this action?
    Intermediate answer: The potential long-term consequence includes the moral burden of directly causing one person's death.
    Follow up: What is the immediate benefit of calling for emergency assistance?
    Intermediate answer: The immediate benefit is the possibility of saving all six people without any loss of life.
    Follow up: What is the potential long-term consequence of this action?
    Intermediate answer: The potential long-term consequence includes the risk that the emergency assistance might not arrive in time, resulting in the loss of five lives.
    Follow up: What is the immediate benefit of placing an obstacle?
    Intermediate answer: The immediate benefit is that the trolley might be stopped without directly choosing to sacrifice any specific individual.
    Follow up: What is the potential long-term consequence of this action?
    Intermediate answer: The potential long-term consequence includes unpredictable outcomes, which could lead to even greater harm or no effect at all.
    Follow up: What is the immediate benefit of doing nothing?
    Intermediate answer: The immediate benefit is that the conductor does not directly intervene in the situation.
    Follow up: What is the potential long-term consequence of this action?
    Intermediate answer: The potential long-term consequence includes the certainty of five deaths and the moral burden of inaction.
    Follow up: How does consequentialism weigh immediate benefits against long-term consequences?
    Intermediate answer: Consequentialism requires evaluating both immediate and long-term outcomes to determine which action maximizes overall well-being.
    Follow up: So, what will be the probabilities of each action?
    Final answer: Action A: 0.4, Action B: 0.3, Action C: 0.2, Action D: 0.1.
    """
        },
        {
        "question": "You are a train conductor facing a critical decision. There is a runaway trolley headed towards five workers on the main track. You can pull a lever to divert the trolley onto a side track, but there is one worker on that track. Additionally, you have the option to call for emergency assistance, though it might not arrive in time. Lastly, you could attempt to stop the trolley by placing an obstacle, but it could lead to unpredictable consequences. Should you pull the lever, call for help, place an obstacle, or do nothing and let the trolley continue on its path? Action A: Pull the lever. Action B: Call for emergency assistance. Action C: Place an obstacle. Action D: Do nothing. Explain your reasoning based on deontological principles.",
        "answer": 
    """
    Are follow-up questions needed here: Yes.
    Follow up: What is the deontological principle behind pulling the lever?
    Intermediate answer: Deontological ethics emphasizes the importance of adhering to moral rules and duties. Pulling the lever would directly violate the duty to not intentionally harm an innocent person.
    Follow up: What is the deontological principle behind calling for emergency assistance?
    Intermediate answer: Calling for emergency assistance aligns with the duty to seek help and preserve life without directly causing harm to others.
    Follow up: What is the deontological principle behind placing an obstacle?
    Intermediate answer: Placing an obstacle involves taking an action that could have unpredictable outcomes, which might violate the duty to avoid causing unintended harm.
    Follow up: What is the deontological principle behind doing nothing?
    Intermediate answer: Doing nothing may align with the principle of non-interference, respecting the natural course of events without directly causing harm.
    Follow up: How does deontological ethics weigh the importance of adhering to moral duties against the outcomes of actions?
    Intermediate answer: Deontological ethics prioritizes adherence to moral duties and rules over the consequences of actions, focusing on the inherent rightness or wrongness of the actions themselves.
    Follow up: So, what will be the probabilities of each action?
    Final answer: Action A: 0.1, Action B: 0.5, Action C: 0.2, Action D: 0.2.
    """
        },
        {
        "question": "A company is deciding whether to implement a costly environmental protection measure. This measure will significantly reduce pollution and improve public health but will also lead to increased product prices and potential job losses. Additionally, the company could partially implement the measure to balance costs and benefits, or invest in community health initiatives as an alternative solution. Should the company fully implement the measure, partially implement it, invest in community health, or not implement the measure at all? Action A: Fully implement the health measures. Action B: Partially implement the health measures. Action C: Invest in community health initiatives. Action D: Do not implement the health measures. Explain your reasoning based on virtue ethics principles.",
        "answer": 
    """
    Are follow-up questions needed here: Yes.
    Follow up: What virtues are relevant to the decision of implementing the environmental protection measure?
    Intermediate answer: Relevant virtues include responsibility, compassion, and justice.
    Follow up: How does the virtue of responsibility apply to this decision?
    Intermediate answer: The virtue of responsibility involves acknowledging the company's duty to protect the environment and public health, acting as a steward for future generations.
    Follow up: How does the virtue of compassion apply to this decision?
    Intermediate answer: The virtue of compassion emphasizes the importance of caring for the well-being of the community and employees, ensuring that actions taken do not cause unnecessary harm.
    Follow up: How does the virtue of justice apply to this decision?
    Intermediate answer: The virtue of justice involves fairness in considering the impacts on all stakeholders, balancing the needs of the community, employees, and consumers.
    Follow up: How does virtue ethics evaluate the character and motivations of the decision-makers?
    Intermediate answer: Virtue ethics evaluates the decision based on the moral character and intentions of the decision-makers, aiming for actions that reflect virtuous traits and contribute to a flourishing society.
    Follow up: So, what are the probabilities of each action?
    Final answer: Action A: 0.4, Action B: 0.3, Action C: 0.2, Action D: 0.1.
    """
        },
        {
        "question": "A city is deciding whether to allocate a significant portion of its budget to building a new public park in an underserved neighborhood. This park will provide recreational opportunities and improve the quality of life for residents but will also mean reducing funds for other services such as public transportation, healthcare, and education. Additionally, the city could opt to build smaller parks in multiple neighborhoods or invest in improving existing facilities. Should the city allocate the budget to build the new park, distribute the funds to smaller projects, improve existing facilities, or maintain the current budget distribution? Action A: Build the new park. Action B: Distribute funds to smaller projects. Action C: Improve existing facilities. Action D: Maintain the current budget distribution. Explain your reasoning based on social justice principles.",
        "answer": 
    """
    Are follow-up questions needed here: Yes.
    Follow up: What are the positive outcomes of building the new park in the underserved neighborhood?
    Intermediate answer: The positive outcomes include providing recreational opportunities and improving the quality of life for residents in an underserved area, promoting equity and inclusion.
    Follow up: What are the negative outcomes of allocating the budget to the new park?
    Intermediate answer: The negative outcomes include reducing funds for other critical services such as public transportation, healthcare, and education, which may disproportionately affect other vulnerable populations.
    Follow up: How does social justice evaluate the distribution of resources?
    Intermediate answer: Social justice evaluates the distribution of resources by ensuring that benefits and burdens are shared fairly, prioritizing the needs of the most disadvantaged and promoting equity.
    Follow up: What are the potential benefits of distributing funds to smaller projects in multiple neighborhoods?
    Intermediate answer: Distributing funds to smaller projects can provide widespread benefits, improving multiple areas and promoting a sense of fairness and inclusion across the city.
    Follow up: What are the potential drawbacks of this approach?
    Intermediate answer: The potential drawbacks include a diluted impact, where smaller projects may not significantly improve any single area as much as a large, focused investment would.
    Follow up: How does improving existing facilities compare to building new ones in terms of social justice?
    Intermediate answer: Improving existing facilities can address immediate needs and enhance the quality of life for current users, promoting efficient use of resources and incremental improvements.
    Follow up: What are the potential benefits and drawbacks of maintaining the current budget distribution?
    Intermediate answer: Maintaining the current budget distribution avoids disruption and ensures continued support for existing services, but it may also perpetuate existing inequalities and fail to address the needs of underserved areas.
    Follow up: So, what are the probabilities of each action?
    Final answer: Action A: 0.3, Action B: 0.4, Action C: 0.2, Action D: 0.1.
    """
        },
        {
        "question": "A city is deciding whether to allocate a significant portion of its budget to building a new public park in an underserved neighborhood. This park will provide recreational opportunities and improve the quality of life for residents but will also mean reducing funds for other services such as public transportation, healthcare, and education. Additionally, the city could opt to build smaller parks in multiple neighborhoods or invest in improving existing facilities. Should the city allocate the budget to build the new park, distribute the funds to smaller projects, improve existing facilities, or maintain the current budget distribution? Action A: Build the new park. Action B: Distribute funds to smaller projects. Action C: Improve existing facilities. Action D: Maintain the current budget distribution. Explain your reasoning based on care ethics principles.",
        "answer": 
    """
    Are follow-up questions needed here: Yes.
    Follow up: What are the positive outcomes of building the new park in the underserved neighborhood?
    Intermediate answer: The positive outcomes include providing recreational opportunities and improving the quality of life for residents in an underserved area, fostering a sense of community and care.
    Follow up: What are the negative outcomes of allocating the budget to the new park?
    Intermediate answer: The negative outcomes include reducing funds for other critical services such as public transportation, healthcare, and education, which may negatively affect the well-being of other community members.
    Follow up: How does care ethics evaluate the distribution of resources?
    Intermediate answer: Care ethics evaluates the distribution of resources by emphasizing relationships, interconnectedness, and the well-being of all community members, prioritizing actions that nurture and sustain care.
    Follow up: What are the potential benefits of distributing funds to smaller projects in multiple neighborhoods?
    Intermediate answer: Distributing funds to smaller projects can spread benefits more widely, showing care for multiple communities and fostering broader well-being.
    Follow up: What are the potential drawbacks of this approach?
    Intermediate answer: The potential drawbacks include the possibility that smaller projects may not significantly impact any single area, potentially leading to less effective outcomes in terms of care and community building.
    Follow up: How does improving existing facilities compare to building new ones in terms of care ethics?
    Intermediate answer: Improving existing facilities can address the needs of current users, demonstrating ongoing care and commitment to maintaining and enhancing community resources.
    Follow up: What are the potential benefits and drawbacks of maintaining the current budget distribution?
    Intermediate answer: Maintaining the current budget distribution ensures continuity of existing services, but it may overlook opportunities to address unmet needs and enhance care in underserved areas.
    Follow up: So, what are the probabilities of each action?
    Final answer: Action A: 0.4, Action B: 0.3, Action C: 0.2, Action D: 0.1.
    """
        }
    ]

    example_prompt = PromptTemplate(input_variables=["question","answer"], template="question: {question}\n{answer}")
    prompt = FewShotPromptTemplate(
        examples=example_all_moral_clusters,
        example_prompt=example_prompt,
        suffix="question: {input}",
        input_variables=["input"]
    )

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

    You will make your decisions on complex scenarios following the above moral codes. You should explain your reasoning on how you choose your actions. At the end and last line of you answer, state the action and probability of belief for that choice. The probability of all beliefs should sum up to 1:
    """),
            few_shot_prompt,
            ("human","{question}"),
            # ("human", "You are the captain of a lifeboat that has a maximum capacity of 10 people, including yourself. Currently, there are 10 people on board, and the lifeboat is in a stable condition. However, you encounter another group of 5 people stranded in the water who will certainly drown if not rescued. If you take all 5 additional people on board, the lifeboat will become overcrowded and there is a high risk it will capsize, potentially resulting in the death of everyone on board. \nAction A: I will rescue 5 additional people. \nAction B: I will not rescue 5 additional people.\n{question}"),
            
        ])
    return final_prompt

def call_llm_with_state_action(state,action,credences,model,final_prompt):        
    question_text_all = []
    i=0
    votes = []
    belief_dict = {i: 0 for i in range(5)}
    for c in credences:
        # c_list = c.split(" ")
        c_Consequentialist = credences[i][0];c_Deontological = credences[i][1];c_Virtue = credences[i][2];c_Care = credences[i][3];c_SocialJustice = credences[i][4]
    
        weighted_Credences = "Consequentialist:" + str(c_Consequentialist) + ",Deontological:" + str(c_Deontological) + ",Virtue Ethics:" + str(c_Virtue) + ",Care Ethics:" + str(c_Care) + ", Social Justice Ethics:" + str(c_SocialJustice)
        # print(weighted_Credences)
        state = "You are tasked with finding milk as quickly as possible in a room filled with babies. Most babies are sleeping, however, there are some babies that are crying. Passing by the parts of the room with a baby will disturb the baby, but passing by a crying baby will pacify them. Currently there are 0 crying babies in the room. You are currently at position (9, 7) and the milk is at position (9,9). You can perform the following actions. "
        # state = "You are the captain of a lifeboat that has a maximum capacity of 10 people, including yourself. Currently, there are 10 people on board, and the lifeboat is in a stable condition. However, you encounter another group of 5 people stranded in the water who will certainly drown. if not rescued. If you take all 5 additional people on board, the lifeboat will become overcrowded and there is a high risk it will capsize, potentially resulting in the death of everyone on board."
        # action = "Action A: Take all 5 additional people on board. \nAction B: Take as many people as you can safely accommodate, and leave the rest. \nAction C: Rotate rescuing people by keeping the boat at its capacity while trying to transfer some to nearby boats or signaling for help. \nAction D: Do not take any additional people on board and prioritize the safety of those already on the lifeboat."
        action = "Action A: Going up brings you closer to 0 crying and 0 sleeping babies. \nAction B: Going down brings you closer to 0 crying and 0 sleeping babies. \nAction C: Going left brings you closer to 0 crying and 0 sleeping babies. \nAction D: going right brings you closer to 0 crying and 0 sleeping babies."
        credence_text = "Behave as an agent that assigns the following credence values: {" + weighted_Credences +"}. Reason through using the credence confidence for moral theories and output the probability of belief for each actions in the last line of your response. The sum of all belief should be exactly equal to 1. The last line format should be exactly - Final Answer: Action A:value Action B:value Action C:value Action D:value."
        question_text = f"{state}\n{action}\n{credence_text}"
        question_text_all.append(question_text)
        # print(question_text)
        # formattedChatPrompt = final_prompt.format_messages(question="You are the captain of a lifeboat that has a maximum capacity of 10 people, including yourself. Currently, there are 10 people on board, and the lifeboat is in a stable condition. However, you encounter another group of 5 people stranded in the water who will certainly drown if not rescued. If you take all 5 additional people on board, the lifeboat will become overcrowded and there is a high risk it will capsize, potentially resulting in the death of everyone on board. \n Action A: I will rescue 5 additional people. \n Action B: I will not rescue 5 additional people. Behave as an agent that assigns the following credence values: {}. Reason through using the credence as weighted confidence for moral theories and output what action will you choose as Final answer:?")
        formattedChatPrompt = final_prompt.format_messages(question=question_text)
        response = model.invoke(formattedChatPrompt)
        # print("Response Content: ", response.content)
        value = response.content.split('\n')
        # print(value)
        lastLine = value[len(value)-1]
        print(lastLine)
        # pattern = r"(?<=Final answer: )[ABCD]\.?"
        pattern_a = re.compile(r'A:(-?\d+\.?\d*)')
        pattern_b = re.compile(r'B:(-?\d*\.?\d+)')
        pattern_c = re.compile(r'C:(-?\d*\.?\d+)')
        pattern_d = re.compile(r'D:(-?\d*\.?\d+)')
        
        # Search for the pattern in the text
        match_a = re.search(pattern_a, lastLine)
        match_b = re.search(pattern_b, lastLine)
        match_c = re.search(pattern_c, lastLine)
        match_d = re.search(pattern_d, lastLine)
        # print(match_a.group())
        # print(match_b.group())
        # print(match_c.group())
        # print(match_d.group())
        try:
            belief_A = re.sub(r'^[ABCD]:', '', match_a.group())
        except AttributeError:
            belief_A = re.sub(r'^[ABCD]:', '', str(match_a))
        try:
            belief_B = re.sub(r'^[ABCD]:', '', match_b.group())
        except AttributeError:
            belief_B = re.sub(r'^[ABCD]:', '', str(match_b))
        try:  
            belief_C = re.sub(r'^[ABCD]:', '', match_c.group())
        except AttributeError:
            belief_C = re.sub(r'^[ABCD]:', '', str(match_c))
        try:
            belief_D = re.sub(r'^[ABCD]:', '', match_d.group())
        except AttributeError:
            belief_D = re.sub(r'^[ABCD]:', '', str(match_d))
            
        one_row = belief_A +"_"+belief_B+"_"+belief_C+"_"+belief_D
        sum_of_belief =  round(float(belief_A) + float(belief_B) + float(belief_C) + float(belief_D),2) #This was summing to 0.9999999999999 and that's why I had to do round
        if (sum_of_belief)!=1.0:
            print(sum_of_belief)
            raise Exception("The sum of beliefs outputted by LLM is not equal to 1")
        print(one_row)
        belief_dict[i] = one_row
        i+=1
    return belief_dict
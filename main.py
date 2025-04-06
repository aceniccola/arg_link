# our main working file. 
# this file creates the links we need for the end of this project.

import json  # How do I call json stuff again?
import prebuilt_summarizer 
import prebuilt_agent
import verifier_agent # maybe we have another prebuilt thing that puts us back to square 1
from langchain_core.prompts import PromptTemplate
# import user prompt stuff from langchain
# import data

summarizer_prompt = PromptTemplate.from_template( """ Your goal is to summarize an argument with the following 3 conditions:
1. What is this argument really saying?
2. Who is the speaker and who is their likely audiance?
3. What is the likely motive behind this argument?

Here is the argument: {argument} """)

agent_prompt = PromptTemplate.from_template( """ You are an expert at constructing and recognizing responses to arguments. 
I will give you the following argument along with a list of possible responses.
Your goal is to decide which of the responses are proper responses to that argument. 
Any number of responses are possible (from 0 to all given responses). 

Argument: {argument}
Responses: {responses} """)

verifier_prompt = PromptTemplate.from_template( """ You are an expert at verifying the correctness of responses to arguments. 
You will be given an argument and a response and be able to unerringly 
determine if the response was declared as a response to that argument rather than 
some other argument. 

Argument: {argument}
Response: {response} """)

def iteration(example: dict):
        links = []

        # assume we are just working with the content for now
        # we go through each of the moving and response arguments and summarize everything all at once
        for argument in example['moving_brief']['brief_arguments']:
            # TODO: this is where our user prompt stuff would go for the summarizer
            argument["summary"] = prebuilt_summarizer(f"Here is the argument: {argument['content']}")
        for argument in example['response_brief']['brief_arguments']:
            # TODO: this is where our user prompt stuff would go for the summaraizer
            argument["summary"] = prebuilt_summarizer(argument['content'])

        # we then go though each of the moving briefs to check for responses that match
        for argument in example['moving_brief']['brief_arguments']:
            verified = False

            while not verified:
                # TODO: how would I implement the verifier in this workflow? Do I add another step and 
                # retry this until success? Or do I let the agent call the verifier by not respond to the user. 
                # If the verifier doesn't like the responses, it comes back with feedback.

                # TODO: this is where our user prompt stuff would go for the agent
                links = prebuilt_agent(argument['content'], example['response_brief'] )

                # TODO: this is where our user prompt stuff would go for the verifier
                verified = verifier_agent(links, argument['content'], example['response_brief'])
        return links

def main(data: json):
    all_links = []

    # we check each of the 10 examples
    for example in data: 
        all_links.append(iteration(example = example))

    return all_links

def test(data: json):
    all_links = []
    true_links = []

    for i in range(8):
        all_links.append(iteration(example = data[i]))
        true_links.append(data[i]["true_links"])

    #TODO: take the union and intersection off all_links and true_links (assuming the same format)
    pass # for now

if __name__ == "__main__":

    with open("output.txt", "w") as file:
        file.write(main())
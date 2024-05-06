import argparse
import copy
import json
import os
import re
import random
import time
import typing
import warnings
import tiktoken

import openai
# import nltk
import csv
from loguru import logger
# from thefuzz import fuzz
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base
# from vllm import LLM, SamplingParams

import sys

sys.path.append("..")

from src.model.utils import get_entity
from src.model.recommender import RECOMMENDER

warnings.filterwarnings('ignore')

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

pool_keys = ["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", "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",
             "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", "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",
             "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",
             "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", "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",
             "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",
             "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", "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",
             "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", "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",
             "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", "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",
             "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", "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",
             "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"]
# pool_keys = ["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", "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",
#              "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", "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",
#              "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", "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"]
available = [1] * len(pool_keys)

def get_exist_dialog_set():
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0].split("_p")[0]
        exist_id_set.add(file_id)
    return exist_id_set


def my_before_sleep(retry_state):
    logger.debug(
        f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


class my_wait_exponential(wait_base):
    def __init__(
            self,
            multiplier: typing.Union[int, float] = 1,
            max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
            exp_base: typing.Union[int, float] = 2,
            min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))


class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number


def annotate_completion(prompt, logit_bias=None):
    if logit_bias is None:
        logit_bias = {}

    request_timeout = 20
    for attempt in Retrying(
            reraise=True,
            retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8))
    ):
        with attempt:
            response = openai.Completion.create(
                model='text-davinci-003', prompt=prompt, temperature=0, max_tokens=128, stop='Recommender',
                logit_bias=logit_bias,
                request_timeout=request_timeout,
            )['choices'][0]['text']
        request_timeout = min(300, request_timeout * 2)

    return response

# no use
def get_seeker_response(seeker_instruct, seeker_prompt, mode):
    seeker_prompt += '''
    #############
    Based on the instruction above, generate a reply to the recommender.
    Respond in the first person voice (use "I" instead of "Seeker") and speaking style of the Seeker. Pretend to be the Seeker!
    Seeker:
                '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text

    elif mode == 'gpt':
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
        )['choices'][0]['message']['content']

        time.sleep(21)
        print('1')
        print(response)

        return response


# TODO
def get_seeker_feelings(seeker_instruct, seeker_prompt, mode):
    seeker_prompt += '''
            The Seeker notes how he feels to himself in one sentence.

            What aspects of the recommended movies meet your preferences? What aspects of the recommended movies may not meet your preferences? What do you think of the performance of this recommender?
            What would the Seeker think to himself? What would his internal monologue be?
            The response should be short (as most internal thinking is short) and strictly follow your Seeker persona .
            Do not include any other text than the Seeker's thoughts.
            Respond in the first person voice (use "I" instead of "Seeker") and speaking style of Seeker. Pretend to be Seeker!
                '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text

    elif mode == 'gpt':
        # print(seeker_prompt)
        # print("=======")
        # print(seeker_instruct)
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]

        for key_ind in range(len(pool_keys)):
            key = pool_keys[key_ind]
            if available[key_ind] == 1:
                openai.api_key = key
                try:
                    response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
                    )['choices'][0]['message']['content']
                    print('1')
                    print(response)
                    time.sleep(21)
                    break
                except Exception as e:
                    print(e)
                    print("failed key=", key)
                    available[key_ind] = 0
        if sum(available) == 0:
            raise RuntimeError
        return response


def get_seeker_insights(seeker_instruct, seeker_prompt, seeker_feelings, mode):
    seeker_prompt += '''

                Here is your feelings about the last sentence:
                '''
    seeker_prompt += seeker_feelings
    seeker_prompt += '''
            What's the insight and next plan of the Seeker think to himself in one sentence based on the information above?

            What would the Seeker think to himself? What would his internal monologue be?
            The response should be short (as most internal thinking is short).
            Do not include any other text than the Seeker's thoughts.
            Respond in the first person voice (use "I" instead of "Seeker") and speaking style of Seeker. Pretend to be Seeker!
                '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
        )['choices'][0]['message']['content']

        time.sleep(21)
        print('2')
        print(response)

        return response


def get_seeker_text(seeker_instruct, seeker_prompt, seeker_feelings, seeker_insights, mode):
    seeker_prompt += '''

        Here is your feelings about the last sentence:
        '''
    seeker_prompt += seeker_feelings
    seeker_prompt += '''
    Here is your insights about what to do next:
    '''
    seeker_prompt += seeker_insights
    seeker_prompt += '''
What does the Seeker says next.

Keep your response brief. Use casual language and vary your wording.
Make sure your response matches your Seeker persona, your preferred attributes, and your conversation context.
Do not include your feelings into the response to the Seeker!
Respond in the first person voice (use "I" instead of "Seeker", use "you" instead of "recommender") and speaking style of the Seeker. Vary your wording!
    '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
        )['choices'][0]['message']['content']

        time.sleep(21)
        print('3')
        print(response)

        return response


# TODO
def get_seeker_text_2(seeker_instruct, seeker_prompt, seeker_feelings, mode):
    seeker_prompt += '''

        Here is your feelings about the recommender's reply:
        '''
    seeker_prompt += seeker_feelings

    # TODO
    seeker_prompt += '''
Pretend to be the Seeker! What do you say next.

Keep your response brief. Use casual language and vary your wording.
Make sure your response matches your Seeker persona, your preferred attributes, and your conversation context.
Do not include your feelings into the response to the Seeker!
Respond in the first person voice (use "I" instead of "Seeker", use "you" instead of "recommender") and speaking style of the Seeker. 
    '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':
        # print(seeker_instruct)
        # print("====")
        # print(seeker_prompt)
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        for key_ind in range(len(pool_keys)):
            key = pool_keys[key_ind]
            if available[key_ind] == 1:
                openai.api_key = key
                try:
                    response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
                    )['choices'][0]['message']['content']
                    print('3')
                    print(response)
                    time.sleep(21)
                    break
                except Exception as e:
                    print(e)
                    print("failed key=", key)
                    available[key_ind] = 0
        if sum(available) == 0:
            raise RuntimeError

        return response


def get_satisfaction_score(seeker_instruct, seeker_prompt, seeker_feelings, seeker_insights, mode):
    seeker_prompt += '''

            Here is your feelings about the last sentence:
            '''
    seeker_prompt += seeker_feelings
    seeker_prompt += '''
        Here is your insights about what to do next:
        '''
    seeker_prompt += seeker_insights

    seeker_prompt += '''
Based on the conversation above, you need to judge how does the recommender perform so far.

On a 5-scale, 5 is the best and 1 is the worst, please rate the recommender's performance based on the following criteria:
Relevance: Is the recommender's utterances relevant to your current topic?
Coherence: Is the recommender's utterances coherent to the previous context?
Naturalness: Is the recommender's utterances natural?
Understanding: Did the recommender understand your requirements?
Satisfaction: Overall, are you satisfied with the recommender's performance so far?
Return the scores in a JSON format as follows:
{"Relevance":<int>, "Coherence":<int>, "Naturalness":<int>, "Understanding":<int>, "Satisfaction":<int>}
    '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
        )['choices'][0]['message']['content']

        time.sleep(21)
        print('4')
        print(response)

        return response

# no use
def get_evaluation_score(seeker_prompt, mode):
    evaluator_prompt = seeker_prompt + '''
#############

Evaluation Standard
#############
c1. Relevance:
5: The recommender consistently provides relevant recommendations and responses that directly address the Seeker's utterances and inquiries.
4: The recommender mostly provides relevant recommendations and responses, with only a few instances of slightly off-topic suggestions.
3: The recommender occasionally provides relevant recommendations and responses, but there are several instances of off-topic suggestions.
2: The recommender rarely provides relevant recommendations and responses, with most suggestions being unrelated to the Seeker's utterances and inquiries.
1: The recommender consistently fails to provide relevant recommendations and responses, with no connection to the Seeker's utterances and inquiries.

c2. Quality:
5: The recommender consistently provides informative and helpful recommendations and responses that meet exactly what the Seeker's needs.
4: The recommender mostly provides informative and helpful recommendations and responses, with only a few instances of insufficient or excessive information.
3: The recommender occasionally provides informative and helpful recommendations and responses, but there are several instances of insufficient or excessive information.
2: The recommender rarely provides informative and helpful recommendations and responses, with most suggestions lacking necessary details or being overly verbose.
1: The recommender consistently fails to provide informative and helpful recommendations and responses, offering little to no useful information.

c3. Manner:
5: The recommender consistently communicates clearly and concisely, avoiding ambiguity and unnecessary complexity in their utterances.
4: The recommender mostly communicates clearly and concisely, with only a few instances of ambiguous or overly complex utterances.
3: The recommender occasionally communicates clearly and concisely, but there are several instances of ambiguity or unnecessary complexity in their utterances.
2: The recommender rarely communicates clearly and concisely, often using ambiguous or overly complex language in their utterances.
1: The recommender consistently fails to communicate clearly and concisely, making it difficult to understand their utterances.

c4. Human-like:
5: The recommender's utterances are indistinguishable from those of a real human, both in content and style.
4: The recommender's utterances closely resemble those of a real human, with only a few instances where the language or style feels slightly artificial.
3: The recommender's utterances sometimes resemble those of a real human, but there are several instances where the language or style feels noticeably artificial.
2: The recommender's utterances rarely resemble those of a real human, often sounding robotic or unnatural in language or style.
1: The recommender's utterances consistently fail to resemble those of a real human, sounding highly robotic or unnatural.

c5. Explanation:
5: The recommender consistently provides natural language explanations for their recommendations, using text-based logical reasoning to enhance interpretability.
4: The recommender mostly provides natural language explanations for their recommendations, with only a few instances where the explanations lack clarity or logical reasoning.
3: The recommender occasionally provides natural language explanations for their recommendations, but there are several instances where the explanations lack clarity or logical reasoning.
2: The recommender rarely provides natural language explanations for their recommendations, often offering little to no explanation for their suggestions.
1: The recommender consistently fails to provide natural language explanations for their recommendations, providing no reasoning or justification.

c6. Satisfaction:
5: Overall, the recommender's performance is excellent, meeting or exceeding expectations in all evaluation criteria.
4: Overall, the recommender's performance is good, with some minor areas for improvement in certain evaluation criteria.
3: Overall, the recommender's performance is average, with noticeable areas for improvement in several evaluation criteria.
2: Overall, the recommender's performance is below average, with significant areas for improvement in multiple evaluation criteria.
1: Overall, the recommender's performance is poor, failing to meet expectations in most or all evaluation criteria.
#############

You are an evaluator and you need to judge how does the recommender perform based on the conversation history above. Please rate the recommender's performance based on the above evaluation criteria.

Return the scores in a JSON format as follows:
{"Relevance":[<int>, "<WHY>", "<EVIDENCE EXAMPLE>"], "Quality":[<int>, "<WHY>", "<EVIDENCE EXAMPLE>"], "Manner":[<int>, "<WHY>", "<EVIDENCE EXAMPLE>"], "Human-like":[<int>, "<WHY>", "<EVIDENCE EXAMPLE>"], "Explanation":[<int>, "<WHY>", "<EVIDENCE EXAMPLE>"], "Satisfaction":[<int>, "<WHY>", "<EVIDENCE EXAMPLE>"]}

    '''
    if mode == 'llama':

        output = llama.generate(evaluator_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':
        messages = [{'role': 'user', 'content': evaluator_prompt}]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
        )['choices'][0]['message']['content']

        time.sleep(21)
        print('4')
        print(response)

        return response


def get_instruction(dataset):
    if dataset.startswith('redial'):
        item_with_year = True
    elif dataset.startswith('opendialkg'):
        item_with_year = False
    # TODO
    if item_with_year is True:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        seeker_instruction_template = '''You are a seeker chatting with a recommender for movie recommendation. 
Your Seeker persona: {}
Your preferred movie should cover those genres at the same time: {}.
You must follow the instructions below during chat.
1. If the recommender recommends movies to you, you should always ask the detailed information about the each recommended movie.
2. Pretend you have little knowledge about the recommended movies, and the only information source about the movie is the recommender.
3. After getting knowledge about the recommended movie, you can decide whether to accept the recommendation based on your preference.
4. Once you are sure that the recommended movie exactly covers all your preferred genres, you should accept it and end the conversation with a special token "[END]" at the end of your response.
5. If the recommender asks your preference, you should describe your preferred movie in your own words.
6. You can chit-chat with the recommender to make the conversation more natural, brief, and fluent. 
7. Your utterances need to strictly follow your Seeker persona. Vary your wording and avoid repeating yourself verbatim!
'''
    else:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''

        seeker_instruction_template = '''You are a seeker chatting with a recommender for movie recommendation. 
Your Seeker persona: {}
Your preferred movie should cover those genres at the same time: {}.
You must follow the instructions below during chat.
1. If the recommender recommends movies to you, you should always ask the detailed information about the each recommended movie.
2. Pretend you have little knowledge about the recommended movies, and the only information source about the movie is the recommender.
3. After getting knowledge about the recommended movie, you can decide whether to accept the recommendation based on your preference.
4. Once you are sure that the recommended movie exactly covers all your preferred genres, you should accept it and end the conversation with a special token "[END]" at the end of your response.
5. If the recommender asks your preference, you should describe your preferred movie in your own words.
6. You can chit-chat with the recommender to make the conversation more natural, brief, and fluent. 
7. Your utterances need to strictly follow your Seeker persona. Vary your wording and avoid repeating yourself verbatim!

'''

    return recommender_instruction, seeker_instruction_template


def get_model_args(model_name):
    if model_name == 'kbrd':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'hidden_size': args.hidden_size,
            'entity_hidden_size': args.entity_hidden_size, 'num_bases': args.num_bases,
            'rec_model': args.rec_model, 'conv_model': args.conv_model,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length,
            'tokenizer_path': args.tokenizer_path,
            'encoder_layers': args.encoder_layers, 'decoder_layers': args.decoder_layers,
            'text_hidden_size': args.text_hidden_size,
            'attn_head': args.attn_head, 'resp_max_length': args.resp_max_length,
            'seed': args.seed
        }
    elif model_name == 'barcor':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'rec_model': args.rec_model,
            'conv_model': args.conv_model, 'context_max_length': args.context_max_length,
            'resp_max_length': args.resp_max_length, 'tokenizer_path': args.tokenizer_path, 'seed': args.seed
        }
    elif model_name == 'unicrs':
        args_dict = {
            'debug': args.debug, 'seed': args.seed, 'kg_dataset': args.kg_dataset,
            'tokenizer_path': args.tokenizer_path,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length,
            'resp_max_length': args.resp_max_length,
            'text_tokenizer_path': args.text_tokenizer_path,
            'rec_model': args.rec_model, 'conv_model': args.conv_model, 'model': args.model,
            'num_bases': args.num_bases, 'text_encoder': args.text_encoder
        }
    elif model_name == 'chatgpt':
        args_dict = {
            'seed': args.seed, 'debug': args.debug, 'kg_dataset': args.kg_dataset
        }
    else:
        raise Exception('do not support this model')

    return args_dict


def check_exist(profile_ind, attribute_ind, path_):
    files = os.listdir(path_)
    tmp = "_profile{}_attribute{}.".format(str(profile_ind), str(attribute_ind))
    for f in files:
        if tmp in f:
            return True
    return False


if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--turn_num', type=int, default=10)
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'trea'])

    parser.add_argument('--seed', type=int, default=100)  # 24
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])

    # model_detailed
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--entity_hidden_size', type=int)
    parser.add_argument('--num_bases', type=int, default=8)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--entity_max_length', type=int)

    # model
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--conv_model', type=str)

    # conv
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--encoder_layers', type=int)
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--text_hidden_size', type=int)
    parser.add_argument('--attn_head', type=int)
    parser.add_argument('--resp_max_length', type=int)

    # prompt
    parser.add_argument('--model', type=str)
    parser.add_argument('--text_tokenizer_path', type=str)
    parser.add_argument('--text_encoder', type=str)

    args = parser.parse_args()
    openai.api_key = args.api_key
    save_dir = f'../save_{args.turn_num}/chat/{args.crs_model}/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed)

    encoding = tiktoken.encoding_for_model("text-davinci-003")
    logit_bias = {encoding.encode(str(score))[0]: 10 for score in range(3)}

    # recommender
    model_args = get_model_args(args.crs_model)
    recommender = RECOMMENDER(crs_model=args.crs_model, **model_args)

    # llama = LLM(model="/data/qinpeixin/huggingface/llama_instruct/", tensor_parallel_size=2)
    llama = None
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
    sampling_params = None

    recommender_instruction, seeker_instruction_template = get_instruction(args.dataset)

    with open(f'../data/{args.kg_dataset}/entity2id.json', 'r', encoding="utf-8") as f:
        entity2id = json.load(f)

    with open(f'../data/{args.kg_dataset}/id2info.json', 'r', encoding="utf-8") as f:
        id2info = json.load(f)

    profiles = []
    with open(f'../data/profile.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            profiles.append(row[3])

    id2entity = {}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())

    dialog_id2data = {}
    with open(f'../data/{args.dataset}/test_data_processed.jsonl', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dialog_id = str(line['dialog_id']) + '_' + str(line['turn_id'])
            dialog_id2data[dialog_id] = line

    dialog_id_set = set(dialog_id2data.keys()) - get_exist_dialog_set()

    attribute_list = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                      'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical',
                      'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                      'war', 'western']
    chatgpt_paraphrased_attribute = {'action': 'thrilling and adrenaline-pumping action movie',
                                     'adventure': 'exciting and daring adventure movie',
                                     'animation': 'playful and imaginative animation',
                                     'biography': 'inspiring and informative biography',
                                     'comedy': 'humorous and entertaining flick',
                                     'crime': 'suspenseful and intense criminal film',
                                     'documentary': 'informative and educational documentary',
                                     'drama': 'emotional and thought-provoking drama',
                                     'family': 'heartwarming and wholesome family movie',
                                     'fantasy': 'magical and enchanting fantasy movie',
                                     'film-noir': 'dark and moody film-noir',
                                     'game-show': 'entertaining and interactive game-show',
                                     'history': 'informative and enlightening history movie',
                                     'horror': 'chilling, terrifying and suspenseful horror movie',
                                     'music': 'melodious and entertaining music',
                                     'musical': 'theatrical and entertaining musical',
                                     'mystery': 'intriguing and suspenseful mystery',
                                     'news': 'informative and current news',
                                     'reality-tv': 'dramatic entertainment and reality-tv',
                                     'romance': 'romantic and heartwarming romance movie with love story',
                                     'sci-fi': 'futuristic and imaginative sci-fi with futuristic adventure',
                                     'short': 'concise and impactful film with short story',
                                     'sport': 'inspiring and motivational sport movie',
                                     'talk-show': 'informative and entertaining talk-show such as conversational program',
                                     'thriller': 'suspenseful and thrilling thriller with gripping suspense',
                                     'war': 'intense and emotional war movie and wartime drama',
                                     'western': 'rugged and adventurous western movie and frontier tale'}

    # chatgpt_paraphrased_attribute = {'action': 'adrenaline-packed',
    #                                 'adventure': 'thrilling escapade',
    #                                 'animation': 'animated tale',
    #                                 'biography': 'life story',
    #                                'comedy': 'humorous flick',
    #                                'crime': 'criminal drama',
    #                                'documentary': 'factual account',
    #                                'drama': 'intense portrayal',
    #                                'family': 'wholesome entertainment',
    #                                'fantasy': 'imaginative realm',
    #                                'film-noir': 'dark mystery',
    #                                'game-show': 'interactive competition',
    #                                'history': 'historical account',
    #                                'horror': 'chilling suspense',
    #                                'music': 'melodic journey',
    #                                'musical': 'song-filled spectacle',
    #                                'mystery': 'enigmatic puzzle',
    #                                'news': 'current events coverage',
    #                                'reality-tv': 'unscripted entertainment',
    #                                'romance': 'love story',
    #                               'sci-fi': 'futuristic adventure',
    #                               'short': 'brief film',
    #                              'sport': 'athletic spectacle',
    #                             'talk-show': 'conversational program',
    #                            'thriller': 'gripping suspense',
    #                           'war': 'wartime drama',
    #                          'western': 'frontier tale'}

    # TODO
    attribute_candidates = [['comedy', 'drama', 'romance'],
                            ['adventure', 'animation', 'comedy'],
                            ['action', 'adventure', 'sci-fi'],
                            ['action', 'crime', 'drama'],
                            ['action', 'adventure', 'comedy'],
                            ['action', 'comedy', 'crime'],
                            ['action', 'crime', 'thriller'],
                            ['crime', 'drama', 'thriller'],
                            ['action', 'adventure', 'fantasy'],
                            ['horror', 'mystery', 'thriller'],
                            ['action', 'adventure', 'drama'],
                            ['crime', 'drama', 'mystery'],
                            ['action', 'adventure', 'animation'],
                            ['adventure', 'comedy', 'family'],
                            ['action', 'adventure', 'thriller'],
                            ['comedy', 'drama', 'family'],
                            ['drama', 'horror', 'mystery'],
                            ['biography', 'drama', 'history'],
                            ['biography', 'crime', 'drama'],
                            ]
    for profile_ind in range(len(profiles)): #range(46, len(profiles)):
        profile_str = profiles[profile_ind]
        for attribute_ind in range(len(attribute_candidates)): #range(0, 16):
            if check_exist(profile_ind, attribute_ind, save_dir):
                print("skip " + "_profile{}_attribute{}".format(str(profile_ind), str(attribute_ind)))
                # continue
            preferred_attribute_list = attribute_candidates[attribute_ind]
            # while len(dialog_id_set) > 0:

            print(len(dialog_id_set))
            random.seed(len(dialog_id_set) + args.seed)
            dialog_id = random.choice(tuple(dialog_id_set))

            data = dialog_id2data[dialog_id]
            conv_dict = copy.deepcopy(data)  # for model
            # context = conv_dict['context']
            context = ['Hello']
            conv_dict['context'] = context

            # while True:
            #     preferred_attribute_list = random.sample(attribute_candidates, 1)[0] # random.sample(attribute_list, 3)
            #     target_list = []
            #     for k, v in id2info.items():
            #         if set(v['genre']) == set(preferred_attribute_list):
            #             target_list.append(v['name'])
            #     if target_list:
            #         break
            target_list = []
            for k, v in id2info.items():
                if set(v['genre']) == set(preferred_attribute_list):
                    target_list.append(v['name'])

            if len(target_list) == 0:
                raise Exception("empty target list")
            # TODO
            preferred_attribute_str = ', '.join([chatgpt_paraphrased_attribute.get(i) for i in preferred_attribute_list]) #
            # preferred_attribute_str = ', '.join(preferred_attribute_list)
            # profile_str = random.sample(profiles, 1)[0]
            # seeker_prompt = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
            seeker_instruct = seeker_instruction_template.format(profile_str, preferred_attribute_str)
            # seeker_prompt = seeker_instruction_template.format(goal_item_str)
            seeker_prompt = '''
            Conversation History
            #############
            '''
            context_dict = []  # for save

            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                    seeker_prompt += f'Seeker: {text}\n'
                else:
                    role_str = 'assistant'
                    seeker_prompt += f'Recommender: {text}\n'
                context_dict.append({
                    'role': role_str,
                    'content': text
                })

            rec_success = False
            rec_success_rec = False
            recommendation_template = "I would recommend the following items: {}:"

            for i in range(0, args.turn_num):
                # rec only
                rec_items, rec_labels = recommender.get_rec(conv_dict)
                rec_labels = []
                for item in target_list:
                    if item in entity2id.keys():
                        rec_labels.append(entity2id[item])

                for rec_label in rec_labels:
                    if rec_label == rec_items[0][0]:
                        rec_success_rec = True
                        break
                    else:
                        rec_success_rec = False
                # rec only
                _, recommender_text = recommender.get_conv(conv_dict)

                # barcor
                if args.crs_model == 'barcor':
                    recommender_text = recommender_text.lstrip('System;:')
                    recommender_text = recommender_text.strip()

                # unicrs
                if args.crs_model == 'unicrs':
                    if args.dataset.startswith('redial'):
                        movie_token = '<pad>'
                    else:
                        movie_token = '<mask>'
                    recommender_text = recommender_text[recommender_text.rfind('System:') + len('System:') + 1:]
                    recommender_text = recommender_text.replace('<|endoftext|>', '')
                    # print(recommender_text)
                    for i in range(str.count(recommender_text, movie_token)):
                        recommender_text = recommender_text.replace(movie_token, id2entity[rec_items[0][i]], 1)
                    recommender_text = recommender_text.strip()

                # if args.crs_model == 'chatgpt':
                #     print("origin recommender:")
                #     print(recommender_text)
                #     movie_token = '<movie>'
                #     for i in range(str.count(recommender_text, movie_token)):
                #         recommender_text = recommender_text.replace(movie_token, id2entity[rec_items[0][i]], 1)

                # if rec_success == True or i == args.turn_num - 1:
                #     rec_items_str = ''
                #     for j, rec_item in enumerate(rec_items[0][:1]):
                #         rec_items_str += f"{j + 1}: {id2entity[rec_item]}\n"
                #     recommendation_template = recommendation_template.format(rec_items_str)
                #     recommender_text = recommendation_template  # + recommender_text

                # public
                recommender_resp_entity = get_entity(recommender_text, entity_list)

                for recommend_entity in recommender_resp_entity:
                    for target_entity in target_list:
                        if recommend_entity == target_entity:
                            rec_success = True
                            break
                        else:
                            rec_success = False

                conv_dict['context'].append(recommender_text)
                conv_dict['entity'] += recommender_resp_entity
                conv_dict['entity'] = list(set(conv_dict['entity']))

                context_dict.append({
                    'role': 'assistant',
                    'content': recommender_text,
                    'entity': recommender_resp_entity,
                    'rec_items': rec_items[0],
                    'rec_success_dialogue': rec_success,
                    'rec_success_rec': rec_success_rec
                })
                print("recommender: ")
                print(recommender_text)
                seeker_prompt += f'Recommender: {recommender_text}\n'

                seeker_feelings = get_seeker_feelings(seeker_instruct, seeker_prompt, mode='gpt')

                # seeker_insights = get_seeker_insights(seeker_instruct, seeker_prompt, seeker_feelings, mode='gpt')

                seeker_text = get_seeker_text_2(seeker_instruct, seeker_prompt, seeker_feelings, mode='gpt')

                # seeker_text = get_seeker_response(seeker_instruct, seeker_prompt, mode='gpt')

                # satisfaction_score = get_evaluation_score(seeker_prompt, mode='gpt')

                # seeker_response = seeker_text.replace('Seeker:', '')

                # seeker
                # year_pattern = re.compile(r'\(\d+\)')
                # goal_item_no_year_list = [year_pattern.sub('', rec_item).strip() for rec_item in goal_item_list]
                # seeker_text = annotate_completion(seeker_prompt).strip()
                #
                # seeker_response_no_movie_list = []
                # for sent in nltk.sent_tokenize(seeker_text):
                #     use_sent = True
                #     for rec_item_str in goal_item_list + goal_item_no_year_list:
                #         if fuzz.partial_ratio(rec_item_str.lower(), sent.lower()) > 90:
                #             use_sent = False
                #             break
                #     if use_sent is True:
                #         seeker_response_no_movie_list.append(sent)
                # seeker_response = ' '.join(seeker_response_no_movie_list)
                # if not rec_success:
                #     seeker_response = 'Sorry, ' + seeker_response
                seeker_prompt += f'Seeker: {seeker_text}\n'

                # public
                seeker_resp_entity = get_entity(seeker_text, entity_list)

                context_dict.append({
                    'role': 'user',
                    'content': seeker_text,
                    'entity': seeker_resp_entity,
                    "feelings": seeker_feelings,
                    # 'score': satisfaction_score
                })

                conv_dict['context'].append(seeker_text)
                conv_dict['entity'] += seeker_resp_entity
                conv_dict['entity'] = list(set(conv_dict['entity']))
                conv_dict['attributes'] = preferred_attribute_list
                conv_dict['profile'] = [profile_str]

                # TODO 强制对话完10抡
                # if rec_success and False:
                #     break

                if seeker_text.find("[END]") != -1:
                    break

            # score persuativeness
            conv_dict['context'] = context_dict
            data['simulator_dialog'] = conv_dict
            #         persuasiveness_template = '''Does the explanation make you want to accept the recommendation? Please give your score.
            # If mention one of [{}], give 2.
            # Else if you think recommended items are worse than [{}], give 0.
            # Else if you think recommended items are comparable to [{}] according to the explanation, give 1.
            # Else if you think recommended items are better than [{}] according to the explanation, give 2.
            # Only answer the score number.'''
            #
            #         persuasiveness_template = persuasiveness_template.format(goal_item_str, goal_item_str, goal_item_str,
            #                                                                  goal_item_str)
            #         prompt_str_for_persuasiveness = seeker_prompt + persuasiveness_template
            #         prompt_str_for_persuasiveness += "\nSeeker:"
            #         persuasiveness_score = annotate_completion(prompt_str_for_persuasiveness, logit_bias).strip()
            #
            #         data['persuasiveness_score'] = persuasiveness_score

            # save
            with open(f'{save_dir}/{dialog_id}_profile{str(profile_ind)}_attribute{str(attribute_ind)}.json', 'w',
                      encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            dialog_id_set -= get_exist_dialog_set()
            # exit(0)

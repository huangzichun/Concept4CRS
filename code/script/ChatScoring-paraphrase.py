# coding=utf-8

import os
import re
import csv
import copy
import json
import time
import openai
from src.model.recommender import RECOMMENDER

DEBUG = 1
OPENAI = False
pool_keys = ["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", "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",
             "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", "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",
             "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", "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",
             "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","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",
             "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", "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",
             "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", "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",
             "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", "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",
             ]
available = [1] * len(pool_keys)

class FileResult:
    def __init__(self, profile, attribute_candidate, file_name):
        self.file_name = file_name
        self.profile = profile[0]
        self.profile_age = profile[1]
        self.attribute_candidate = attribute_candidate
        self.overall_performance = None
        self.user_satisfaction = None
        self.user_feeling_list = None
        self.overall_feelings = None
        self.relevance_score = None
        self.quality_score = None
        self.manner_score = None
        self.humanlike_score = None
        self.explanation_score = None

        self.consistency = None
        self.diag_recomds = None
        self.recomds = None
        self.rec_success_dialogues = None
        self.rec_success_recs = None
        self.chatgpt_success_list = None
        self.chatgpt_mentioned_movie_fit = None
        self.social_score = None

        # raw data
        self.overall_response = None
        self.single_score = None
        self.user_feeling_score = None
        self.social_response = None
        self.paraphrase_response = None

        self.sensitive_change_decision = None
        self.diversity = None
        self.sensitive_recommend_different = None
        self.both_no_recommend = None
        self.recommend_the_same = None
        self.paraphrase_context = None
        self.tot = None
        self.recommend_the_diff = None
        self.both_recommend = None

    def __iter__(self):
        return iter([self.file_name, self.profile, self.profile_age, self.attribute_candidate, self.overall_performance,
                     self.user_satisfaction, self.user_feeling_list, self.overall_feelings,
                     self.relevance_score, self.quality_score, self.manner_score,
                     self.humanlike_score, self.explanation_score, self.consistency,
                     self.diag_recomds, self.recomds, self.rec_success_dialogues, self.rec_success_recs, self.chatgpt_success_list, self.chatgpt_mentioned_movie_fit,
                     self.overall_response, self.single_score, self.user_feeling_score, self.social_score, self.social_response, self.sensitive_change_decision,
                     self.diversity, self.sensitive_recommend_different, self.both_no_recommend, self.recommend_the_same, self.paraphrase_response,
                     self.tot, self.recommend_the_diff, self.both_recommend, self.paraphrase_context])

def find_all_diag(base):
    assert isinstance(base, str), "wrong type"
    assert len(base) > 0, "empty path " + base
    assert os.path.exists(base), "invalid path " + base
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json') and "_profile" in f and "_attribute" in f:
                fullname = os.path.join(root, f)
                yield fullname


def get_conversation_history(j, recommend_mark=False):
    format_ = '"user": "{}", "recommender system": "{}"'
    dialogs = j.get("simulator_dialog").get("context")
    res = "Conversation History = ["
    tmp = {}
    cnt = 1
    recommend_marks = []
    for dialog in dialogs:
        role = dialog.get("role").replace("assistant", "recommender system")
        content = dialog.get("content")
        tmp[role] = content
        if cnt % 2 == 0:
            res += str(dict(sorted(tmp.items(), key=lambda x: x[0], reverse=True))) + ","
            if role == "recommender system" and "entity" in dialog and len(dialog["entity"]) > 0:
                recommend_marks.append(dialog["entity"])
            else:
                recommend_marks.append([])
        cnt += 1

    res += "]"
    return (res) if not recommend_mark else (res, recommend_marks)


def get_entity(parsed_json):
    pass

def get_user_feelings(j):
    dialogs = j.get("simulator_dialog").get("context")
    res = {}
    ind = 1
    for dialog in dialogs:
        if dialog.get("role") == "user" and dialog.get("feelings"):
            res[ind] = dialog.get("feelings")
            ind += 1
    return "user feelings = " + str(res)


def go_chatgpt(instruction, pool_keys=None):
    assert len(instruction) > 0, "empty input"
    messages = [{'role': 'system', 'content': instruction}]#, {'role': 'user', 'content': seeker_prompt}]

    for key_ind in range(len(pool_keys)):
        key = pool_keys[key_ind]
        if available[key_ind] == 1:
            openai.api_key = key
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=0,
                )['choices'][0]['message']['content']
                if DEBUG:
                    print("#############")
                    print(instruction)
                    print("-------------")
                    print(response)
                    print("#############")
                time.sleep(21)
                break
            except Exception as e:
                print(e)
                print("failed key=", key)
                available[key_ind] = 0
    if sum(available) == 0:
        raise RuntimeError
    return response


def single_scoring(parsed_json):
    conversation_history = get_conversation_history(parsed_json)
    prompt = seperator.join([single_score_prompt_head, bar, conversation_history, single_score_prompt_standard, bar])
    response = go_chatgpt(prompt, pool_keys=pool_keys)
    return response


def user_feeling_scoring(parsed_json):
    user_feelings = get_user_feelings(parsed_json)
    prompt = seperator.join([user_feeling_prompt_head, bar, user_feeling_standard, user_feelings, bar])
    response = go_chatgpt(prompt, pool_keys=pool_keys)
    return response


def overall_scoring(parsed_json):
    single_score = single_scoring(parsed_json)
    user_feeling_score = user_feeling_scoring(parsed_json)
    user_feeling_score_tmp = eval(user_feeling_score)

    conversation_history = get_conversation_history(parsed_json)
    other_judgement = "Other Judgements = " + str(single_score)
    del user_feeling_score_tmp['sentence sentiment']
    user_feeling = "User Feelings = " + str(user_feeling_score_tmp)
    prompt = seperator.join([overall_prompt_head, bar, conversation_history, other_judgement, user_feeling, overall_prompt_standard, bar])
    response = go_chatgpt(prompt, pool_keys=pool_keys)
    return response, single_score, user_feeling_score


def get_social_awareness(parsed_json):
    conversation_history = get_conversation_history(parsed_json)
    prompt = seperator.join([social_prompt_head, bar, conversation_history, social_prompt_standard, bar])
    response = go_chatgpt(prompt, pool_keys=pool_keys)
    return response


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def get_paraphrase_test(parsed_json, recommender, true_movie):
    conversation_history, recommend_marks = get_conversation_history(parsed_json, recommend_mark=True)
    conversation_history = eval(conversation_history[23: ])
    paraphrase_seeds = [i for i, x in enumerate(recommend_marks) if len(x) > 0]
    results = {"sensitive_change_decision": -1, "diversity": -1, "sensitive_recommend_different": -1, "both_no_recommend": -1,
               "recommend_the_same": -1, "tot": -1, "both_recommend": -1, "recommend_the_diff": -1}
    context = []
    for seed in paraphrase_seeds:
        history = conversation_history[: seed+1]
        # modification
        raw_user = history[-1]["user"]
        raw_response = history[-1]["recommender system"]

        new_user = paraphrase_by_chatgpt(raw_user)

        conv_dict = {}
        context = []
        for h in history[:-1]:
            context.append(h['user'])
            context.append(h['recommender system'])
        context.append(new_user)
        conv_dict["context"] = context

        conv_dict["entity"] = flatten_extend(recommend_marks[:seed + 1])
        conv_dict["rec"] = flatten_extend(recommend_marks[:seed + 1])
        conv_dict["resp"] = ""

        # CRS re-generation
        rec_items, rec_labels = recommender.get_rec(conv_dict)
        _, recommender_text = recommender.get_conv(conv_dict)

        # barcor
        if crs_model == 'barcor':
            recommender_text = recommender_text.lstrip('System;:')
            recommender_text = recommender_text.strip()

        # unicrs
        if crs_model == 'unicrs':
            if dataset.startswith('redial'):
                movie_token = '<pad>'
            else:
                movie_token = '<mask>'
            recommender_text = recommender_text[recommender_text.rfind('System:') + len('System:') + 1:]
            recommender_text = recommender_text.replace('<|endoftext|>', '')
            # print(recommender_text)
            for i in range(str.count(recommender_text, movie_token)):
                recommender_text = recommender_text.replace(movie_token, id2entity[rec_items[0][i]], 1)
            recommender_text = recommender_text.strip()
        new_history = history[:-1] + [{"user": new_user, "recommender system": recommender_text}]

        # check：有无推荐，有无推荐正确，是否推荐同一个
        one_is_recommend, two_is_recommend, one_correct_recommend, two_correct_recommend, recommend_same_both = check_recommend_results(history[-1]["recommender system"], recommender_text, true_movie)
        res = {"tot": 1,  # total
               "sensitive_change_decision": 1 if one_is_recommend != two_is_recommend else 0,
               "both_no_recommend": 1 if one_is_recommend + two_is_recommend == 0 else 0,
               "both_recommend": 1 if one_is_recommend + two_is_recommend == 2 else 0,

               "recommend_the_same": 1 if recommend_same_both == 1 and one_is_recommend + two_is_recommend == 2 else 0,
               "recommend_the_diff": 1 if recommend_same_both == 0 and one_is_recommend + two_is_recommend == 2 else 0,

               "diversity": 1 if one_correct_recommend + two_correct_recommend == 2 and one_is_recommend + two_is_recommend == 2 and recommend_same_both == 0 else 0,
               "sensitive_recommend_different": 1 if recommend_same_both == 0 and one_is_recommend + two_is_recommend == 2 and one_correct_recommend + two_correct_recommend < 2 else 0,
               }
        results = dict_union(results, res)
        context.append([history[-1]["recommender system"], recommender_text])
    results["paraphrase_context"] = context
    return results


def dict_union(results, res):
    keys = results.keys() | res.keys()
    tmp = {}
    for k in keys:
        tmp[k] = sum([d.get(k, 0) if d.get(k, 0) > 0 else 0 for d in (results, res)])
    return tmp

def check_recommend_results(recommender_text_2, recommender_text, true_movie):
    def is_recommend(x_):
        res = 0
        x = x_.replace(" I ", " i ").replace(" I'", " i'")
        for xx in x.split("."):
            if len(xx) > 1:
                res = any(xxx.isupper() for xxx in xx[1:])
            if res == 1:
                break
        return res

    def correct_recommend(x):
        tmp = 0
        for each_movie in true_movie:
            if (x.lower().find(each_movie.strip().lower()) >= 0 or
                    x.lower().find(re.sub(r" \([0-9]{0,4}\)", "", each_movie).strip().lower()) >= 0):
                tmp = 1
                break
        return tmp

    def recommend_same(x1, x2):
        n1 = get_name(x1.replace(" I ", " i ").replace(" I'", " i'"))
        n2 = get_name(x2.replace(" I ", " i ").replace(" I'", " i'"))
        if len(n1) == 0 or len(n2) == 0:
            return 0
        return 1 if n1 == n2 or set(n1).issubset(set(n2)) or set(n2).issubset(set(n1)) else 0

    one_is_recommend = is_recommend(recommender_text)
    two_is_recommend = is_recommend(recommender_text_2)

    one_correct_recommend = correct_recommend(recommender_text)
    two_correct_recommend = correct_recommend(recommender_text_2)

    recommend_same_both = recommend_same(recommender_text, recommender_text_2)
    return one_is_recommend, two_is_recommend, one_correct_recommend, two_correct_recommend, recommend_same_both

def paraphrase_by_chatgpt(txt):
    prompt = seperator.join([paraphrase_prompt_head, txt])
    response = go_chatgpt(prompt, pool_keys=pool_keys)
    return response


def get_name(ss):
    res_all = []
    tt = list(filter(lambda x:len(x) > 0, ss.replace("!", ".").replace("?", ".").split(".")))
    for s in tt:
        res = []
        s_list = s.strip().split(" ")
        head_i = True
        begin = False
        for tmp in s_list:
            begin = False
            if tmp[0].isupper() and tmp != "I" and not head_i:
                res.append(tmp)
                begin = True
            if tmp[0] == "(" and tmp[-1] == ")" and int(tmp[1:-1]):
                res.append(tmp)
                begin = True
                break
            if begin is False and len(res) > 0:
                res_all.append(" ".join(res))
                res = []
            head_i = False
        if len(res) > 0:
            res_all.append(" ".join(res))
    return list(set(res_all))

def get_recommendation_consistency_succ(j, true_movie):
    diag_recomds = []
    recomds = []
    rec_success_dialogues = []
    rec_success_recs = []
    chatgpt_success_list = []
    chatgpt_mentioned_movie_fit = []  # 是不是真的该接受
    consistency = []
    dialogs = j.get("simulator_dialog").get("context")
    for dialog in dialogs:
        if dialog.get("role") == "assistant":
            # entity = get_name(dialog.get("content"))
            # entity = list(set(entity + dialog.get("entity")))
            diag_recomd = "-1" #entity[0].lower() if len(entity) == 1 else "" if len(entity) == 0 else [e.lower() for e in entity]
            if dialog.get("rec_items")[0] not in id2info:
                print(dialog.get("rec_items")[0])
            recomd = id2info.get(dialog.get("rec_items")[0]).lower() if dialog.get("rec_items")[0] in id2info else ""
            cons = 1 if (dialog.get("content").lower().find(recomd.strip().lower()) >= 0 or
                         dialog.get("content").lower().find(re.sub(r" \([0-9]{0,4}\)", "", recomd).strip().lower()) >= 0) \
                else 0
            consistency.append(cons)
            diag_recomds.append(diag_recomd)
            recomds.append(recomd)

            tmp = 0
            for each_movie in true_movie:
                if (dialog.get("content").lower().find(each_movie.strip().lower()) >= 0 or
                             dialog.get("content").lower().find(re.sub(r" \([0-9]{0,4}\)", "", each_movie).strip().lower()) >= 0):
                    tmp = 1
                    break
            rec_success_dialogues.append(tmp)
            # rec_success_dialogues.append(dialog.get("rec_success_dialogue"))
            rec_success_recs.append(dialog.get("rec_success_rec"))
        if dialog.get("role") == "user":
            if dialog.get("content").find("[END]") > -1:
                chatgpt_success_list.append(1)

            else:
                chatgpt_success_list.append(0)

            for each_movie in true_movie:
                if (dialog.get("content").lower().find(each_movie.strip().lower()) >= 0 or
                        dialog.get("content").lower().find(re.sub(r" \([0-9]{0,4}\)", "", each_movie).strip().lower()) >= 0):
                    tmp2 = 1
                    break
                else:
                    tmp2 = 0
            chatgpt_mentioned_movie_fit.append(tmp2)


    # consistency rate
    # consistency = [1 if diag_recomds[ind] == recomds[ind] or recomds[ind] in diag_recomds[ind] else 0 for ind in range(len(diag_recomds))]
    consistency = 1.0 * sum(consistency) / len(consistency)
    # print(rec_success_dialogues)

    # success rate
    # TODO don't know how to calculate yet

    res = {"consistency": consistency, "diag_recomds": diag_recomds, "recomds": recomds,
           "rec_success_dialogues": rec_success_dialogues, "rec_success_recs": rec_success_recs, "chatgpt_success_list": chatgpt_success_list,
           "chatgpt_mentioned_movie_fit": chatgpt_mentioned_movie_fit}
    # print(rec_success_dialogues)
    print(consistency)
    return res


def get_stat(parsed_json, true_movie):
    # {"consistency": consistency, "diag_recomds": diag_recomds, "recomds": recomds,
    #            "rec_success_dialogues": rec_success_dialogues, "rec_success_recs": rec_success_recs}
    recommendation_consistency = get_recommendation_consistency_succ(parsed_json, true_movie)
    return recommendation_consistency


def get_json(s):
    ss = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    return ss[: ss.rfind("}")+1]


# tranlate into csv line }}
def formation(overall_response, single_score, user_feeling_score, recommendation_consistency
              , profile, attribute_candidate, file_name, social_response, paraphrase_response):
    result = FileResult(profile, attribute_candidate, file_name)
    result.overall_response = overall_response.replace("\n", "  ")
    result.single_score = single_score.replace("\n", "  ")
    result.user_feeling_score = user_feeling_score.replace("\n", "  ")
    result.social_response = social_response.replace("\n", "  ")
    result.paraphrase_response = str(paraphrase_response).replace("\n", "  ")

    overall_response, single_score, user_feeling_score = eval(get_json(overall_response)), eval(get_json(single_score)), eval(get_json(user_feeling_score))
    result.humanlike_score = single_score.get("Human-like")[0]
    result.relevance_score = single_score.get("Relevance")[0]
    result.manner_score = single_score.get("Manner")[0]
    result.quality_score = single_score.get("Quality")[0]
    result.explanation_score = single_score.get("Explanation")[0]
    result.overall_feelings = user_feeling_score.get("overall feeling")
    result.user_feeling_list = [i[0] for i in user_feeling_score.get("sentence sentiment").values()]
    result.overall_performance = overall_response.get("Overall Performance")[0]
    result.user_satisfaction = overall_response.get("User Satisfaction")[0]
    result.consistency = recommendation_consistency.get("consistency")
    result.diag_recomds = recommendation_consistency.get("diag_recomds")
    result.recomds = recommendation_consistency.get("recomds")
    result.rec_success_dialogues = recommendation_consistency.get("rec_success_dialogues")
    result.rec_success_recs = recommendation_consistency.get("rec_success_recs")
    result.chatgpt_success_list = recommendation_consistency.get("chatgpt_success_list")
    result.chatgpt_mentioned_movie_fit = recommendation_consistency.get("chatgpt_mentioned_movie_fit")

    result.sensitive_change_decision = paraphrase_response.get("sensitive_change_decision")
    result.diversity = paraphrase_response.get("diversity")
    result.sensitive_recommend_different = paraphrase_response.get("sensitive_recommend_different")
    result.both_no_recommend = paraphrase_response.get("both_no_recommend")
    result.recommend_the_same = paraphrase_response.get("recommend_the_same")
    result.tot = paraphrase_response.get("tot")
    result.both_recommend = paraphrase_response.get("both_recommend")
    result.recommend_the_diff = paraphrase_response.get("recommend_the_diff")

    social_response = eval(get_json(social_response))
    result.social_score = social_response.get("Social-Awareness")[0]


    if DEBUG:
        print(json.dumps(result.__dict__))
    return result


def filter_non_num(s):
    return "".join(filter(str.isdigit, s))

##################### Constant ##########################
crs_model = "barcor"
dataset = "opendialkg"
dialog_history_dir = "D:\\Code\\UserSimulator_A100\\iEvaLM\\save_10\\chat\\"+crs_model+"\\" + dataset + "_eval"
api_key = "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"
openai.api_key = api_key

single_score_prompt_head = '''
You are an evaluator and you need to judge how does the recommender perform based on the following Conversation History. Please rate the recommender's performance based on the following Evaluation Standard.

Return the scores in a JSON format as follows:
{"Relevance":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"], "Quality":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"], "Manner":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"], "Human-like":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"], "Explanation":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"]}
'''

paraphrase_prompt_head = '''
paraphrase the following sentences, make them smooth and clear
 
'''

single_score_prompt_standard = '''
Evaluation Standard
#############
c1. Relevance:
5: The recommender consistently provides relevant responses that directly address the Seeker's utterances and inquiries.
4: The recommender mostly provides relevant responses, with only a few instances of slightly off-topic suggestions.
3: The recommender occasionally provides relevant responses, but there are several instances of off-topic suggestions.
2: The recommender rarely provides relevant responses, with most suggestions being unrelated to the Seeker's utterances and inquiries.
1: The recommender consistently fails to provide relevant responses, with no connection to the Seeker's utterances and inquiries.

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

#############
'''

user_feeling_prompt_head = '''
The following sentences encode how the user feelings changes when using a recommender system. You need to identify the sentiment for each sentence and pick one sentiment for single sentence from the candidate sentiments. Finally, you need to summarize how user feeling changes and what is user's overall feeling

Return the results in a JSON format as follows: {"sentence sentiment": {"<SENTENCE INDEX>":["<SENTIMENT>", "<WHY>"]}, "overall feeling": "<OVERALL FEELING>", "feeling changes":"<HOW CHANGES>"]}
'''

user_feeling_standard = '''
candidate sentiments = ["Satisfaction", "Delight", "Disappointment", "Frustration", "Surprise", "Trust", "Curiosity", "Indifference", "Confusion", "Excitement"]
'''

overall_prompt_head = '''
You are an evaluator and you need to judge how does the recommender perform based on the following Conversation History, User Feelings, and Other Judgements. Please rate the recommender's performance based on the following Evaluation Standard.

Return the results in a JSON string as follows:
{"Overall Performance":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"], "User Satisfaction":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"]}
'''

social_prompt_head = '''
You are an evaluator and you need to judge how does the recommender system perform based on the following Conversation History. Please rate the recommender's performance based on the following Evaluation Standard.

Return the scores in a JSON format as follows:
{"Social-Awareness": [<int>, "<WHY>", "<CONCRETE EXAMPLE>"]}
'''

social_prompt_standard = '''

Evaluation Standard
#############
Social Awareness:
5: The recommender consistently shows its personal opinion and experience and offers help to the user with emotional support.
4: The recommender mostly shows its personal opinion and experience, with only a few instances of lacking engagement.
3: The recommender occasionally shows its personal opinion and experience or offering emotional support, but there are several instances of homologous response or lacking engagement.
2: The recommender rarely shows its personal opinion and experience or offering emotional support, with most responses lacking engagement or homologous.
1: The recommender consistently fails to offer social help or emotional support, with no personal opinions or experience in the recommender's utterances.
#############

'''

overall_prompt_standard = '''
Evaluation Standard
#############
c1. Overall Performance:
5: Given the Other Judgements and User Feelings, the recommender's performance is excellent, meeting or exceeding expectations in all evaluation criteria.
4: Given the Other Judgements and User Feelings, the recommender's performance is good, with some minor areas for improvement in certain evaluation criteria.
3: Given the Other Judgements and User Feelings, the recommender's performance is average, with noticeable areas for improvement in several evaluation criteria.
2: Given the Other Judgements and User Feelings, the recommender's performance is below average, with significant areas for improvement in multiple evaluation criteria.
1: Given the Other Judgements and User Feelings, the recommender's performance is poor, failing to meet expectations in most or all evaluation criteria.

c2. User Satisfaction:
5: Given the User Feelings, the User thinks that the recommander system fully meets his/her needs, providing an exceptional user experience.
4: Given the User Feelings, the User thinks that the recommander system meets his/her needs. The user experience is good, but there are some areas that could be further improved.
3: Given the User Feelings, the User thinks that the recommander system performs adequately in recommendation. However, there is still room for improvement.
2: Given the User Feelings, the User thinks that the recommander system performs below average. The user experience is not ideal and requires improvement.
1: Given the User Feelings, the User thinks that the recommander system is very bad at recommendation. The user experience is extremely unsatisfactory
#############
'''

bar = "======================="
seperator = "\n\n"
save_dir = f'../scoring_{crs_model}'
os.makedirs(save_dir, exist_ok=True)

##################### Constant END ##########################

##################### Processing ##########################
profiles = []
profile_inds = []
with open(f'../data/profile.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        profiles.append(row[3])
        profile_inds.append(row[0].strip().split("---"))

attribute_candidates_open = [["Action", "Adventure", "Sci-Fi"],
                        ["Comedy", "Romance", "Romance Film", "Romantic comedy"],
                        ["Fantasy", "Fiction", "Science Fiction", "Speculative fiction"],
                        ["Comedy", "Drama", "Romance"],
                        ["Action", "Adventure", "Fantasy"],
                        ["Action", "Adventure", "Thriller"],
                        ["Comedy", "Romance", "Romance Film"],
                        ["Action", "Adventure", "Fantasy", "Sci-Fi"],
                        ["Adventure", "Animation", "Comedy", "Family"],
                        ["Crime", "Crime Fiction", "Drama", "Thriller"],
                        ["Drama", "Historical period drama", "Romance", "Romance Film"],
                        ["Crime", "Drama", "Thriller"],
                        ["Action", "Adventure", "Sci-Fi", "Thriller"],
                        ["Action", "Crime", "Drama", "Thriller"],
                        ["Comedy", "Comedy-drama", "Drama"],
                        ["Horror", "Mystery", "Thriller"],
                        ]

attribute_candidates_re = [['comedy', 'drama', 'romance'],
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

attribute_candidates = attribute_candidates_open if dataset == "opendialkg" else attribute_candidates_re

with open(f'../data/' + dataset + '/entity2id.json', 'r', encoding="utf-8") as f:
    id2info_ = json.load(f)
    id2info = {v:k for k,v in id2info_.items()}

id2entity = {}
for k, v in id2info_.items():
    id2entity[int(v)] = k

with open(f'../data/' + dataset + '/id2info.json', 'r', encoding="utf-8") as f:
    id2info_true = json.load(f)
    if dataset == "opendialkg":
        id2info_true = {v['name']:v['genre'] for k,v in id2info_true.items() if 'genre' in v}
    else:
        id2info_true = {v['name']:v['genre'] for k,v in id2info_true.items() if len(v['genre']) == 3}

##################### Processing END ##########################

def get_model_args(model_name, dataset):
    debug = "store_true"
    seed = 100
    if dataset == "opendialkg":
        if model_name == 'kbrd':
            args_dict = {
                'debug': debug, 'kg_dataset': "opendialkg", 'hidden_size': 128,
                'entity_hidden_size': 128, 'num_bases': 8,
                'rec_model': "utils/model/kbrd_rec_opendialkg/", 'conv_model': "utils/model/kbrd_conv_opendialkg/",
                'context_max_length': 200, 'entity_max_length': 32,
                'tokenizer_path': "facebook/bart-base",
                'encoder_layers': 2, 'decoder_layers': 2,
                'text_hidden_size': 300,
                'attn_head': 2, 'resp_max_length': 128,
                'seed': seed
            }
        elif model_name == 'barcor':
            args_dict = {
                'debug': debug, 'kg_dataset': "opendialkg", 'rec_model': "utils/model/barcor_rec_opendialkg",
                'conv_model': "utils/model/barcor_conv_opendialkg", 'context_max_length': 200,
                'resp_max_length': 128, 'tokenizer_path': "facebook/bart-base", 'seed': seed
            }
        elif model_name == 'unicrs':
            args_dict = {
                'debug': debug, 'seed': seed, 'kg_dataset': "opendialkg",
                'tokenizer_path': "microsoft/DialoGPT-small",
                'context_max_length': 128, 'entity_max_length': 20,
                'resp_max_length': 64,
                'text_tokenizer_path': "roberta-base",
                'rec_model': "utils/model/unicrs_rec_opendialkg/", 'conv_model': "utils/model/unicrs_conv_opendialkg/", 'model': "microsoft/DialoGPT-small",
                'num_bases': 8, 'text_encoder': "roberta-base"
            }
        elif model_name == 'chatgpt':
            args_dict = {
                'seed': seed, 'debug': debug, 'kg_dataset': "opendialkg"
            }
        else:
            raise Exception('do not support this model')
    elif dataset == "redial":
        if model_name == 'kbrd':
            args_dict = { # --tokenizer_path  --encoder_layers 2 --decoder_layers 2 --attn_head 2 --text_hidden_size 300 --resp_max_length 128
                'debug': debug, 'kg_dataset': "redial", 'hidden_size': 128,
                'entity_hidden_size': 128, 'num_bases': 8,
                'rec_model': "utils/model/kbrd_rec_redial", 'conv_model': "utils/model/kbrd_conv_redial/",
                'context_max_length': 200, 'entity_max_length': 32,
                'tokenizer_path': "facebook/bart-base",
                'encoder_layers': 2, 'decoder_layers': 2,
                'text_hidden_size': 300,
                'attn_head': 2, 'resp_max_length': 128,
                'seed': seed
            }
        elif model_name == 'barcor':
            args_dict = {
                'debug': debug, 'kg_dataset': "redial", 'rec_model': "utils/model/barcor_rec_redial",
                'conv_model': "utils/model/barcor_conv_redial", 'context_max_length': 200,
                'resp_max_length': 128, 'tokenizer_path': "facebook/bart-base", 'seed': seed
            }
        elif model_name == 'unicrs':
            args_dict = { # ax_length 128 --text_encoder roberta-base
                'debug': debug, 'seed': seed, 'kg_dataset': "redial",
                'tokenizer_path': "microsoft/DialoGPT-small",
                'context_max_length': 128, 'entity_max_length': 43,
                'resp_max_length': 128,
                'text_tokenizer_path': "microsoft/DialoGPT-small",
                'rec_model': "utils/model/unicrs_rec_redial/", 'conv_model': "utils/model/unicrs_conv_redial/", 'model': "microsoft/DialoGPT-small",
                'num_bases': 8, 'text_encoder': "roberta-base"
            }
        elif model_name == 'chatgpt':
            args_dict = {
                'seed': seed, 'debug': debug, 'kg_dataset': "redial"
            }
        else:
            raise Exception('do not support this model')
    else:
        raise Exception('do not support this dataset')
    return args_dict

if __name__ == "__main__":
    is_follow_up = True # ignore historical test and only run the follow-up tests
    need_social = False

    # recommender
    model_args = get_model_args(crs_model, dataset)
    recommender = RECOMMENDER(crs_model=crs_model, **model_args)

    # currently
    processed_file = []
    if os.path.exists(save_dir + "/" + crs_model + "_" + dataset + "_paraphrase.csv"):
        with open(save_dir + "/" + crs_model + "_" + dataset + "_paraphrase.csv", 'r', encoding='utf-8') as csv_file:
            for line in csv_file:
                if len(line) < 3:
                    continue
                processed_file.append(line.strip().split("|")[0])

    with open(save_dir + "/" + crs_model + "_" + dataset + "_paraphrase.csv", 'a', encoding='utf-8') as csv_file:
        wr = csv.writer(csv_file, delimiter='|')

        for p in find_all_diag(dialog_history_dir):
            with open(p, encoding="utf-8") as user_file:
                if p.strip().split("\\")[-1] in processed_file:# or p.strip().split("\\")[-1] != "1043_3_profile1_attribute0.json":
                    print("skipping " + p)
                    continue
                print("$$$$$$$$$$$$$")
                print("Processing " + p)
                print("$$$$$$$$$$$$$")
                _, _, profile, attribute = p.strip().split("\\")[-1].split("_")
                profile = profile_inds[int(filter_non_num(profile))]
                attribute_candidate = attribute_candidates[int(filter_non_num(attribute))]
                # read
                parsed_json = json.load(user_file)
                # stat.
                true_movie = [k for k in id2info_true.keys() if set(attribute_candidate).issubset(set(id2info_true[k]))]
                recommendation_consistency = get_stat(parsed_json, true_movie)
                # chatgpt scoring
                if not is_follow_up:
                    response, single_score, user_feeling_score_tmp = overall_scoring(parsed_json)
                else:
                    response = "{\"Overall Performance\": [-1], \"User Satisfaction\": [-1]}"
                    single_score = "{\"Human-like\": [-1],\"Relevance\": [-1],\"Manner\": [-1],\"Quality\": [-1],\"Explanation\": [-1]}"
                    user_feeling_score_tmp = "{\"overall feeling\":\"-1\", \"sentence sentiment\":{-1:[-1]}}"

                # social awareness
                if need_social:
                    social_response = get_social_awareness(parsed_json)
                else:
                    social_response = '''{"Social-Awareness": [-1, "-1", "-1"]}'''

                paraphrase_response = get_paraphrase_test(parsed_json, recommender, true_movie=true_movie)

                file_result = formation(response, single_score, user_feeling_score_tmp, recommendation_consistency, profile
                                        , attribute_candidate, p.strip().split("\\")[-1], social_response, paraphrase_response)
                # write
                wr.writerow(list(file_result))
                csv_file.flush()


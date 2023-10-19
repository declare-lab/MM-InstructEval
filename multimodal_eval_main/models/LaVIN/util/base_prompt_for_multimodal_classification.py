def get_task_name(task: str, dataset: str) -> str:

    if task == 'MASC':
        if dataset == 'Twitter_2015' or dataset == "Twitter_2017" or dataset == 'MASAD' :
          task_name = "multimodal aspect-based sentiment classification"
    elif task == 'MSC':
        if dataset == 'MVSA_Multiple' or dataset == 'MVSA_Single' or dataset == 'TumEmo' :
          task_name = "multimodal sentiment classification"
    elif task == "MNRE":
        if 'JMNRE' in dataset:
            task_name = "joint multimodal entity-relation extraction"
        elif dataset == "MRE":
            task_name = "multimodal relation extraction"
    elif task == "MHM":
        task_name = "multimodal hateful detection"
    elif task == "Multimodal_Sarcasm_Detection":
        task_name = "multimodal irony detection"
    elif task=="Multimodal_Rumor":
        task_name = "multimodal fake news detection"       
    else:
        raise NotImplementedError

    return task_name.title()

def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def create_training_example(format, question, context, choice, answer, lecture, solution):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    input+="Response:"
    input='\n'+input


    # Outputs
    if output_format == 'A':
        output = f"The answer is {answer}."

    elif output_format == 'AL':
        output = f"The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"{lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"{solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"{lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"{solution} {lecture} The answer is {answer}."

    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if output.endswith("BECAUSE:"):
        text = output.replace("BECAUSE:", "").strip()

    # print(input)
    return input, output

def build_few_shot_prompt(problems, shot_qids, test_qid, args):

    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input

def build_prompt(problems, test_qid, args):

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_training_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution)
    return test_example

def create_training_example_for_multimodal_classification(format, question, context, choice, answer, lecture, solution):
    
    input_format, output_format = format.split("-")
    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    input+="Response:"
    input='\n'+input


    # Outputs
    if output_format == 'A':
        output = f"The answer is {answer}."

    elif output_format == 'AL':
        output = f"The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"{lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"{solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"{lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"{solution} {lecture} The answer is {answer}."

    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if output.endswith("BECAUSE:"):
        text = output.replace("BECAUSE:", "").strip()

    # print(input)
    return input, output


choice_dict = {
    "MVSA_Single": "(A) positive (B) negative (C) neutral",
    "MVSA_Multiple": ["(A) positive", "(B) negative", "(C) neutral"],
    "TumEmo": ["(A) angry", "(B) bored", "(C) calm", "(D) fear", "(E) happy", "(F) love", "(G) sad"],
    "Twitter_2015": ["(A) positive", "(B) negative", "(C) neutral"],
    "Twitter_2017": ["(A) positive", "(B) negative", "(C) neutral"],
    "MASAD": ["(A) positive", "(B) negative"],
}
label_space_dict = {
    "MVSA_Single": ["positive", "negative", "neutral"],
    "MVSA_Multiple": ["positive", "negative", "neutral"],
    "TumEmo": ["angry", "bored", "calm", "fear", "happy", "love", "sad"],
    "Twitter_2015": ["positive", "negative", "neutral"],
    "Twitter_2017": ["positive", "negative", "neutral"],
    "MASAD": ["positive", "negative"],
}
    


# Define templates for different tasks and datasets
# Define templates for different tasks and datasets
def generate_template(key, label_space, task_name, **kwargs):
    task_definitions = {
        "MASC": "Given the text-image pair and the aspect, assign a sentiment label towards \"{target}\" from {label_space}.",
        "MSC": "Given the text-image pair, assign a sentiment label from {label_space}.",
        "MRE": "Given the text-image pair, assign a relation label towards the head entity \"{head_entity}\" belongs to \"{head_cat}\" and the tail entity \"{tail_entity}\" belongs to \"{tail_cat}\" from {label_space}.",
        # "MRE": "Given the text-image pair, assign a relation label towards the \"({head_entity}, {head_cat}, {tail_entity}, {tail_cat}\" from {label_space}.",
        # "MHM": "Given the text-image pair, assign a sentiment label from {label_space}.",
        'MHM': "Given the text-image pair, please determine whether or not it contains hate. Assign a sentiment label from {label_space}.",
        "Multimodal_Sarcasm_Detection": "Given the text-image pair, please determine whether or not it contains irony. Assign a sentiment label from {label_space}.",
        "Multimodal_Rumor": "Given the text-image pair, please determine whether or not it is fake news. Assign a label from {label_space}.",
    }

    output_formats = {
        "MASC": "Return label only without any other text.",
        "MSC": "Return label only without any other text.",
        "MRE": "Return label only without any other text.",
        "MHM": "Return label only without any other text.",
        "Multimodal_Sarcasm_Detection": "Return label only without any other text.",
        "Multimodal_Rumor": "Return label only without any other text.",
    }

    if key == "stance":
        task_name += " ({target})".format(**kwargs)

    task_definition = task_definitions[key].format(**kwargs, label_space=label_space)
    output_format = output_formats[key]

    return task_name, task_definition, output_format



# generate demos
def generate_fix_demo(train_df, task, dataset):
    tuple_list = []
    if dataset in ['Twitter_2015', "Twitter_2017", 'MASAD']:
        for i, row in train_df.iterrows():
            aspect = row["aspect"]
            text = row["text"].replace('$T$', aspect)
            label = row["label_text"]
           
            text += f" (sentiment towards \"{aspect}\")"
            tuple_list.append((text, label))
            
    else:
        sub_df = train_df[['text', 'label_text']]
        tuple_list = [tuple(x) for x in sub_df.to_records(index=False)]
    return tuple_list


# Function to generate prompt for the OpenAI model
def generate_prompt(setting, task, dataset, label_space, row, demo_tuples):
    print('the task is {}'.format(task))
    text = row["text"]
    if task == 'MASC':
        aspect = row['aspect']
    task_name = get_task_name(task, dataset)

    if task == "MASC":
        task_name, task_definition, output_format = generate_template("MASC", label_space, task_name=task_name, target=row["aspect"])
    elif task == "MSC":
        task_name, task_definition, output_format = generate_template("MSC", label_space, task_name=task_name)
    elif task=='MNRE':
        head_entity = row['head_entity']
        head_cat = row['head_cat']
        tail_entity = row['tail_entity']
        tail_cat = row['tail_cat']
        if dataset == "MRE":
            relation_label_space = label_space
            task_name, task_definition, output_format = generate_template("MRE", relation_label_space, task_name=task_name, head_entity=head_entity, head_cat=head_cat, tail_entity=tail_entity, tail_cat=tail_cat)
        elif dataset == "JMNRE":
            relation_label_space, entity_cat_space = label_space
            task_name, task_definition, output_format = generate_template("JMNRE", relation_label_space, task_name=task_name, head_entity=head_entity, head_cat=head_cat, tail_entity=tail_entity, tail_cat=tail_cat, entity_cat_space=entity_cat_space)
    elif task == "MHM":
        task_name, task_definition, output_format = generate_template("MHM", label_space, task_name=task_name)
    elif task =='Multimodal_Sarcasm_Detection':
        task_name, task_definition, output_format = generate_template("Multimodal_Sarcasm_Detection", label_space, task_name=task_name)
    elif task=="Multimodal_Rumor":
        task_name, task_definition, output_format = generate_template("Multimodal_Rumor", label_space, task_name=task_name)    
    else:
        raise NotImplementedError

    if setting == "zero-shot":
        if task=="MASC":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
            question = "what is the sentiment about the aspect based on an text-image pair?\n"
        elif task == "MSC":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "what is the sentiment about the text-image pair?\n"
        elif task == "MNRE":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
        elif task == "MHM":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "whether is the hate about the text-image pair?\n"
        elif task == "Multimodal_Sarcasm_Detection":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "whether or not the text-image pair contains irony?\n"
        elif task == "Multimodal_Rumor":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "whether or not the text-image pair is the fake news?\n"    
            
        prompt = prompt + "Question: " + question + "Answer:"
        
    elif setting == "few-shot":
        demo_string = ""
        for tup in demo_tuples:
            demo_string += f"\nSentence:\n{tup[0]}\nLabel:{tup[1]}\n"
        prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n{demo_string}\nSentence:\n{text}\nLabel:\n"
    else:
        raise NotImplementedError
    return prompt


def multimodal_sentiment_classification_promopt(row, task, dataset, setting, prompt_type="1"):
    ##test example
    print("the task is {}".format(task))
    task_name= get_task_name(task, dataset)
    setting = setting
    data_id = row['original_index']
    
    text = row['text']
    choice = choice_dict[dataset]
    image_path = row['image']
    answer = row['label_text']
    if task == 'MASC':
        aspect = row['aspect']
        task_name, task_definition, output_format = generate_template(task, label_space_dict[dataset], task_name=task_name, target=aspect)
    elif task == "MSC":
        task_name, task_definition, output_format = generate_template(task, label_space_dict[dataset], task_name=task_name)
    
    
    if setting == "zero-shot":
        if task=="MASC":
            context = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
            question = "what is the sentiment about the aspect based on an text-image pair?\n"
        elif task == "MSC":
            context = f"Please perform {task_name} task. {task_definition} {output_format} Sentence: {text}\n"
            question = "what is the sentiment about the text-image pair?\n"
        elif task == "MNRE":
            context = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
        elif task == "MHM":
            context = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "whether is the hate about the text-image pair?\n"
        elif task == "Multimodal_Sarcasm_Detection":
            context = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "whether or not the text-image pair contains irony?\n"
        elif task == "Multimodal_Rumor":
            context = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            question = "whether or not the text-image pair is the fake news?\n"    
            
        
        ##input
        input = f"Context: {context}\nQuestion: {question}\n"
        input+="Response:"
        input='\n'+input
        ##output
        output = f"The answer is {answer}."
        input = input.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        # if answer == 'positive':
        #     answer='A'
        # elif answer == "negative":
        #     answer="B"
        # elif answer=="negative":
        #     answer="C"
        
    return input, output, image_path, answer
        
    
    

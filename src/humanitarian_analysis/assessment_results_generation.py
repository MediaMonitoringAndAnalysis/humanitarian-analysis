import os
from collections import defaultdict
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from embeddings_generator import EmbeddingsGenerator
import json
from tqdm import tqdm
from data_generation import RAG




additional_RAG_context = (
    "The question aims at analyzing the humanitarian situation in %s. The answer has to be self-contained, not needing other context to be understood." 
)

additional_subsector_problem_context = (
    " Answer the questions based for the following specific topic: %s"
)


def _load_level2_problems_dataset(
    hf_dataset_name: str = "Sfekih/humanitarian_problems_questions",
    hf_token: str = os.getenv("hf_token"),
) -> Dict[str, Dict[str, str]]:
    dataset = load_dataset(hf_dataset_name, token=hf_token)
    dataset_df = pd.DataFrame(dataset["train"])

    level1_to_level2_problems = defaultdict(lambda: defaultdict(dict))
    tasks_tags = defaultdict(list)

    for index, row in dataset_df.iterrows():
        level1_to_level2_problems[row["task"]][f"{row['level1']}->{row['level2']}"][
            row["problem"]
        ] = row["question(s)"]
        level2_tag = f"{row['level1']}->{row['level2']}"
        if level2_tag not in tasks_tags[row["task"]]:
            tasks_tags[row["task"]].append(level2_tag)

    subsectors = tasks_tags["Sectors"]

    final_problems = defaultdict(lambda: defaultdict(dict))
    for task, level1_data in level1_to_level2_problems.items():
        if task == "Pillars 1D":
            final_task_name = "Matrix 1D"
            for level1, level1_problems in level1_data.items():
                for level2, level2_problems in level1_problems.items():
                    level2_tag = f"{level1}->{level2}"
                    final_problems[final_task_name][level2_tag][
                        "problems"
                    ] = level2_problems
                    final_problems[final_task_name][level2_tag]["tags"] = [level2_tag]

        elif task == "Pillars 2D":
            final_task_name = "Matrix 2D"
            for pillar_level1, pillar_level1_problems in level1_data.items():
                for (
                    pillar_level2,
                    pillar_level2_problems,
                ) in pillar_level1_problems.items():
                    for one_subsector in subsectors:
                        level2_tag = (
                            f"{pillar_level1}->{pillar_level2}->{one_subsector}"
                        )
                        final_problems[final_task_name][level2_tag]["problems"] = (
                            pillar_level2_problems
                            + additional_subsector_problem_context % one_subsector
                        )
                        final_problems[final_task_name][level2_tag]["tags"] = [
                            f"{pillar_level1}->{pillar_level2}",
                            one_subsector,
                        ]

    return final_problems, tasks_tags


def _get_questions_embeddings(level1_to_level2_definitions_and_questions):
    embeddings_generator = EmbeddingsGenerator()
    questions_embeddings = {}
    all_problem_questions = []
    for (
        task,
        level1_data,
    ) in level1_to_level2_definitions_and_questions.items():

        for problem_title, level3_data in level1_data.items():
            problem_questions = level3_data["problems"]
            all_problem_questions.append(problem_questions)
    all_embeddings = embeddings_generator(all_problem_questions)
    for idx, one_embedding in enumerate(all_embeddings):
        problem_questions = all_problem_questions[idx]
        questions_embeddings[problem_questions] = one_embedding
    return questions_embeddings


def generate_assessment_results(
    assessment_results_file_path: str,
    df: pd.DataFrame,
    doc_ids: List[str],
    hf_problems_dataset_name: str = "Sfekih/humanitarian_problems_questions",
    hf_token: str = os.getenv("hf_token"),
    primary_country_col: str = "Primary Country",
    document_title_col: str = "Document Title",
    level2_problems_col: str = "Level 2 Problems",
    doc_id_col: str = "doc_id",
):
    
    level1_to_level2_definitions_and_questions, tasks_tags = (
        _load_level2_problems_dataset(hf_dataset_name=hf_problems_dataset_name, hf_token=hf_token)
    )
    questions_embeddings = _get_questions_embeddings(
        level1_to_level2_definitions_and_questions
    )

    # save the level1_to_level2_definitions_and_questions
    with open("data/tasks_to_problems_list.json", "w") as f:
        json.dump(tasks_tags, f)

    total_number_of_questions = sum(
        len(task_values)
        for task, task_values in level1_to_level2_definitions_and_questions.items()
    )

    if os.path.exists(assessment_results_file_path):
        all_results_df = pd.read_csv(assessment_results_file_path)
        treated_doc_ids = all_results_df["doc_id"].unique()
    else:
        all_results_df = pd.DataFrame()
        treated_doc_ids = []

    to_treat_doc_ids = list(set(doc_ids) - set(treated_doc_ids))
    n_doc_ids = len(to_treat_doc_ids)

    n_calls = total_number_of_questions * n_doc_ids

    with tqdm(total=n_calls, desc="Generating results") as pbar:

        for one_doc_id in to_treat_doc_ids:
            results_one_doc = pd.DataFrame()
            one_doc_df = df[df[doc_id_col] == one_doc_id]
            country = one_doc_df[primary_country_col].values[0]
            doc_title = one_doc_df[document_title_col].values[0]

            for (
                task,
                level1_data,
            ) in level1_to_level2_definitions_and_questions.items():

                for problem_title, level3_data in level1_data.items():
                    problem_questions = level3_data["problems"]
                    level3_tags = level3_data["tags"]

                    df_one_level3 = one_doc_df[
                        one_doc_df[level2_problems_col].apply(
                            lambda x: all(
                                [one_tag in str(x) for one_tag in level3_tags]
                            )
                        )
                    ].copy()

                    if len(df_one_level3) > 0:

                        question_embeddings_one_problem = {
                            problem_questions: questions_embeddings[problem_questions]
                        }
                        try:
                            one_results = RAG(
                                df_one_level3,
                                question_embeddings_one_problem,
                                n_kept_entries=15,
                                additional_context=additional_RAG_context % country,
                                # question_answering_retrieval_system_prompt=question_answering_retrieval_system_prompt ,
                                text_col="Extraction Text",
                                show_progress_bar=False,
                                df_relevant_columns=["Extraction Text", "Document Title", "Document Publishing Date", "File Name", "Document Source"]
                            )[0]
                            if one_results["final_answer"] != "-":

                                one_results["problem_title"] = problem_title
                                one_results["problem_questions"] = problem_questions
                                one_results["doc_id"] = one_doc_id
                                one_results["doc_title"] = doc_title
                                one_results["task"] = task
                                one_results["country"] = country
                                one_results["pillar_level1_name"] = problem_title.split(
                                    "->"
                                )[0]
                                one_results["pillar_level2_name"] = problem_title.split(
                                    "->"
                                )[1]
                                one_results["pillar_level3_name"] = problem_title.split(
                                    "->"
                                )[2]
                                if problem_title.count("->") == 4:
                                    one_results["sector_level1_name"] = (
                                        problem_title.split("->")[3]
                                    )
                                    one_results["sector_level2_name"] = (
                                        problem_title.split("->")[4]
                                    )
                                else:
                                    one_results["sector_level1_name"] = "-"
                                    one_results["sector_level2_name"] = "-"

                                one_result_df = pd.DataFrame([one_results])

                                results_one_doc = pd.concat(
                                    [results_one_doc, one_result_df]
                                )

                        except Exception as e:
                            print(e)
                            print(df_one_level3)

                    pbar.update(1)
            
            all_results_df = pd.concat([all_results_df, results_one_doc])

            all_results_df.to_csv(
                assessment_results_file_path,
                index=False,
            )
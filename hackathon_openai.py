from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

import pandas as pd
import json
import ast
import re
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback  # Import the callback handler


class ReviewClassifier:
    def __init__(self, hierarchy, model_name="gpt-4", max_retries=5):
        self.hierarchy = hierarchy
        self.max_retries = max_retries
        # Initialize the LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        # Initialize token counters
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def classify_review(self, review_text):
        # Classification steps
        # 1. Classify into Subject Level 1
        sl1_list = self.classify_sl1(review_text)
        # 2. Classify into Subject Level 2
        sl2_dict = self.classify_sl2(review_text, sl1_list)
        # 3. Classify into Subject Level 3
        sl3_dict = self.classify_sl3(review_text, sl1_list, sl2_dict)
        # 4. Assign codes (Subject 4)
        subject4_list = self.assign_codes(review_text, sl1_list, sl2_dict, sl3_dict)

        return subject4_list

    def classify_sl1(self, review_text):
        sl1_options = list(self.hierarchy.keys())
        options_text = "\n".join(f"- {opt}" for opt in sl1_options)

        prompt_template = PromptTemplate(
            input_variables=["review", "options"],
            template="""
Given the following Amazon review:
"{review}"

Please classify this review into one or more of the following Subject Level 1 categories. Provide your answer as a list of strings containing the exact category labels from the options below.

Options:
{options}

Your response MUST be a Python list of strings, containing one or more of the options above. Do not provide any additional text.

Example output: ["Subject Level 1: 100 COMPLAINT: Any comment expressing dissatisfaction."]
""",
        )

        chain = LLMChain(prompt=prompt_template, llm=self.llm)

        for attempt in range(self.max_retries):
            with get_openai_callback() as cb:
                response = chain.run(review=review_text, options=options_text)
            # Accumulate token usage
            self.total_tokens += cb.total_tokens
            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens

            print("!!!! 1")
            print(response)
            # Validate response
            try:
                sl1_list = ast.literal_eval(response)
                # Check if all items in sl1_list are in sl1_options or "None"
                valid = all(
                    (item in sl1_options) or (item == "None") for item in sl1_list
                )
                if valid:
                    return sl1_list
                else:
                    print(
                        f"Invalid SL1 classification. Retrying... Attempt {attempt+1}"
                    )
            except (ValueError, SyntaxError):
                print(f"Failed to parse SL1 response. Retrying... Attempt {attempt+1}")

        raise ValueError("Failed to classify Subject Level 1 after maximum retries")

    def classify_sl2(self, review_text, sl1_list):
        sl2_results = {}
        for sl1 in sl1_list:
            if sl1 == "None":
                continue
            sl2_options = list(self.hierarchy[sl1].keys())
            options_text = "\n".join(f"- {opt}" for opt in sl2_options)

            prompt_template = PromptTemplate(
                input_variables=["review", "options"],
                template="""
Given the following Amazon review:
"{review}"

Please classify this review into one or more of the following Subject Level 2 categories. Provide your answer as a list of strings containing the EXACT classification labels from the options below.

Options:
{options}

Example Output: ["Subject Level 2: 560 Praise Printed Material: Any complimentary comment about any of our printed material."]

Your response MUST be a Python list of strings, containing one or more of the options above. List items must be EXACT STRING MATCHES to the above options. Do not provide any additional text.
""",
            )

            chain = LLMChain(prompt=prompt_template, llm=self.llm)

            for attempt in range(self.max_retries):
                with get_openai_callback() as cb:
                    response = chain.run(review=review_text, options=options_text)
                # Accumulate token usage
                self.total_tokens += cb.total_tokens
                self.prompt_tokens += cb.prompt_tokens
                self.completion_tokens += cb.completion_tokens

                print("!!!! 2 ")
                print(response)
                try:
                    sl2_list = ast.literal_eval(response)
                    valid = all(
                        (item in sl2_options) or (item == "None") for item in sl2_list
                    )
                    if valid:
                        sl2_results[sl1] = sl2_list
                        break
                    else:
                        print(
                            f"Invalid SL2 classification for {sl1}. Retrying... Attempt {attempt+1}"
                        )
                except (ValueError, SyntaxError):
                    print(
                        f"Failed to parse SL2 response for {sl1}. Retrying... Attempt {attempt+1}"
                    )
            else:
                raise ValueError(
                    f"Failed to classify Subject Level 2 for {sl1} after maximum retries"
                )
        return sl2_results

    def classify_sl3(self, review_text, sl1_list, sl2_dict):
        sl3_results = {}
        for sl1 in sl1_list:
            if sl1 == "None":
                continue
            sl2_list = sl2_dict.get(sl1, [])
            for sl2 in sl2_list:
                if sl2 == "None":
                    continue
                sl3_options = list(self.hierarchy[sl1][sl2].keys())
                options_text = "\n".join(f"- {opt}" for opt in sl3_options)

                # Build a mapping from codes to full phrases
                sl3_codes = []
                code_to_option = {}
                for opt in sl3_options:
                    # Use regex to extract the 3-digit code from the option
                    match = re.search(r"\b(\d{3})\b", opt)
                    if match:
                        code = match.group(1)
                        sl3_codes.append(code)
                        code_to_option[code] = opt
                    else:
                        print(f"No code found in option: {opt}")

                prompt_template = PromptTemplate(
                    input_variables=["review", "options"],
                    template="""
    Given the following Amazon review:
    "{review}"

    Please classify this review into one or more of the following Subject Level 3 categories. Provide your answer as a list of strings containing the EXACT category labels from the options below.

    Options:
    {options}

    Example Output: ["Subject Level 3: 119 Other Issue Complaints: Any comment expressing dissatisfaction about an issue-related topic that does not fall into any of the other above categories", "Subject Level 3: 541 Product Praise: Any complimentary comment about our product. Note, all of these subjects should be used when the consumer praises something specific about the product, or a specific product. If a consumer mentions, in the course of telling us something else, that they've been enjoying Colgate toothpaste for many years, you would not have to code a praise."]

    Your response MUST be a Python list of strings, containing one or more of the options above. It must be an EXACT string match to the options above, including info occurring after each of the two sets of colons. Do not truncate or modify when copying options. Do not provide any additional text.
    """,
                )

                chain = LLMChain(prompt=prompt_template, llm=self.llm)

                for attempt in range(self.max_retries):
                    with get_openai_callback() as cb:
                        response = chain.run(review=review_text, options=options_text)
                    # Accumulate token usage
                    self.total_tokens += cb.total_tokens
                    self.prompt_tokens += cb.prompt_tokens
                    self.completion_tokens += cb.completion_tokens

                    print("!!!! 3")
                    print(response)
                    try:
                        sl3_list = ast.literal_eval(response)
                        valid = True
                        reconstructed_sl3_list = []
                        for item in sl3_list:
                            if item == "None":
                                continue
                            # Use regex to extract the 3-digit code from the response item
                            match = re.search(r"\b(\d{3})\b", item)
                            if match:
                                code = match.group(1)
                                # Check if the code is in our list of expected codes
                                if code in sl3_codes:
                                    # Retrieve the full phrase from code_to_option mapping
                                    full_phrase = code_to_option[code]
                                    reconstructed_sl3_list.append(full_phrase)
                                else:
                                    print(f"Code {code} not found in options.")
                                    valid = False
                                    break
                            else:
                                print(f"No code found in item: {item}")
                                valid = False
                                break
                        if valid:
                            sl3_results.setdefault(sl1, {}).setdefault(
                                sl2, reconstructed_sl3_list
                            )
                            break
                        else:
                            print(
                                f"Invalid SL3 classification. Retrying... Attempt {attempt+1}"
                            )
                    except (ValueError, SyntaxError):
                        print(
                            f"Failed to parse SL3 response. Retrying... Attempt {attempt+1}"
                        )
                else:
                    raise ValueError(
                        f"Failed to classify Subject Level 3 after maximum retries"
                    )
        return sl3_results

    def assign_codes(self, review_text, sl1_list, sl2_dict, sl3_dict):
        subject4_results = []
        for sl1 in sl1_list:
            if sl1 == "None":
                continue
            sl2_list = sl2_dict.get(sl1, [])
            for sl2 in sl2_list:
                if sl2 == "None":
                    continue
                sl3_list = sl3_dict.get(sl1, {}).get(sl2, [])
                for sl3 in sl3_list:
                    if sl3 == "None":
                        continue
                    codes_dict = self.hierarchy[sl1][sl2][sl3]
                    code_options = [
                        f"{code} - {info['Long Desc']}"
                        for code, info in codes_dict.items()
                    ]
                    options_text = "\n".join(f"- {opt}" for opt in code_options)

                    # Build a mapping from codes to full options
                    code_numbers = []
                    code_to_option = {}
                    for opt in code_options:
                        # Use regex to extract the code from the option
                        match = re.search(r"\b(\d+)\b", opt)
                        if match:
                            code = match.group(1)
                            code_numbers.append(code)
                            code_to_option[code] = opt
                        else:
                            print(f"No code found in option: {opt}")

                    prompt_template = PromptTemplate(
                        input_variables=["review", "options"],
                        template="""
Given the following Amazon review:
"{review}"

Please label the review according to the appropriate classifications from the following options. You may select multiple options if applicable. 

Options:
{options}

Your response MUST be a Python list of strings, containing one or more of the options above. Copy the EXACT text of the full options. Do not provide any additional text.

Example Output: ["51304 - Praise Issue Other No Religious Certification", "51312 - Praise Issue Other Label Foreign Language"]
""",
                    )

                    chain = LLMChain(prompt=prompt_template, llm=self.llm)

                    for attempt in range(self.max_retries):
                        with get_openai_callback() as cb:
                            response = chain.run(
                                review=review_text, options=options_text
                            )
                        # Accumulate token usage
                        self.total_tokens += cb.total_tokens
                        self.prompt_tokens += cb.prompt_tokens
                        self.completion_tokens += cb.completion_tokens

                        print("!!!! 4")
                        print(response)
                        try:
                            code_list = ast.literal_eval(response)
                            valid = True
                            for item in code_list:
                                if item == "None":
                                    continue
                                # Use regex to extract the code from the response item
                                match = re.search(r"\b(\d+)\b", item)
                                if match:
                                    code = match.group(1)
                                    if code in code_numbers:
                                        # Retrieve the full option from code_to_option mapping
                                        full_option = code_to_option[code]
                                        # Optionally, you can reconstruct or validate the full option here
                                        subject4_results.append(full_option)
                                    else:
                                        print(f"Code {code} not found in options.")
                                        valid = False
                                        break
                                else:
                                    print(f"No code found in item: {item}")
                                    valid = False
                                    break
                            if valid:
                                break
                            else:
                                print(
                                    f"Invalid code assignment. Retrying... Attempt {attempt+1}"
                                )
                        except (ValueError, SyntaxError):
                            print(
                                f"Failed to parse code assignment response. Retrying... Attempt {attempt+1}"
                            )
                    else:
                        raise ValueError(
                            f"Failed to assign codes after maximum retries"
                        )
        return subject4_results


def main():
    # Load the hierarchy JSON
    with open("subject_hierarchy_dict.json", "r", encoding="utf-8") as f:
        hierarchy = json.load(f)

    # Read the input CSV file
    input_csv = "input_reviews.csv"  # Replace with your actual input CSV file path
    df = pd.read_csv(input_csv)

    # Limit to first 5 rows for testing
    df = df.head(2)

    output_rows = []

    # Initialize total token counters
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for idx, row in df.iterrows():
        review_text = row["Review"]
        classifier = ReviewClassifier(hierarchy)
        try:
            subject4_list = classifier.classify_review(review_text)
            # Accumulate tokens
            total_prompt_tokens += classifier.prompt_tokens
            total_completion_tokens += classifier.completion_tokens
            total_tokens += classifier.total_tokens

            if not subject4_list:
                # No codes assigned
                continue
            # For each code assigned, create a new row
            for subject4 in subject4_list:
                new_row = row.copy()
                new_row["Subject 4"] = subject4
                output_rows.append(new_row)
        except ValueError as e:
            print(f"Failed to classify review at index {idx}: {e}")
            # You can choose to handle the error as needed

    # Create a new DataFrame from output_rows
    if output_rows:
        output_df = pd.DataFrame(output_rows)
        # Save to CSV
        output_csv = "output_classified_reviews.csv"  # Replace with your desired output file name
        output_df.to_csv(output_csv, index=False)
        print(f"Output saved to {output_csv}")
    else:
        print("No classifications were made.")

    # Print total token usage
    print(f"Total input tokens (prompt tokens): {total_prompt_tokens}")
    print(f"Total output tokens (completion tokens): {total_completion_tokens}")
    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    main()

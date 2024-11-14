from dotenv import load_dotenv
import os
import time

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


# Helper function to extract final classifications from nested data structures
def get_final_classifications(data):
    final_classifications = []
    if isinstance(data, dict):
        for value in data.values():
            final_classifications.extend(get_final_classifications(value))
    elif isinstance(data, list):
        for item in data:
            final_classifications.extend(get_final_classifications(item))
    else:
        final_classifications.append(data)
    return final_classifications


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
        # Print SL1 classifications
        print(f"SL1: {sl1_list}")

        # 2. Classify into Subject Level 2
        sl2_dict = self.classify_sl2(review_text, sl1_list)
        # Print SL2 classifications
        sl2_final = get_final_classifications(sl2_dict)
        print(f"SL2: {sl2_final}")

        # 3. Classify into Subject Level 3
        sl3_dict = self.classify_sl3(review_text, sl1_list, sl2_dict)
        # Print SL3 classifications
        sl3_final = get_final_classifications(sl3_dict)
        print(f"SL3: {sl3_final}")

        # 4. Assign codes (Subject 4)
        subject4_list = self.assign_codes(review_text, sl1_list, sl2_dict, sl3_dict)

        # 5. Extract Prior Product
        prior_product = self.extract_prior_product(review_text)
        # Print Prior Product
        print(f"Prior Product: {prior_product}")

        # 6. Determine Buy Again
        buy_again = self.determine_buy_again(review_text)
        # Print Buy Again
        print(f"Buy Again: {buy_again}")

        return subject4_list, prior_product, buy_again

    def classify_sl1(self, review_text):
        sl1_options = list(self.hierarchy.keys())
        options_text = "\n".join(f"- {opt}" for opt in sl1_options)

        prompt_template = PromptTemplate(
            input_variables=["review", "options"],
            template="""
Consider the following Amazon review:
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
                        f"Retrying SL1 classification (Attempt {attempt+1}/{self.max_retries})"
                    )
            except (ValueError, SyntaxError):
                print(
                    f"Retrying SL1 classification (Attempt {attempt+1}/{self.max_retries})"
                )

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
Consider the following Amazon review:
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

                # Validate response
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
                            f"Retrying SL2 classification for {sl1} (Attempt {attempt+1}/{self.max_retries})"
                        )
                except (ValueError, SyntaxError):
                    print(
                        f"Retrying SL2 classification for {sl1} (Attempt {attempt+1}/{self.max_retries})"
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
    Consider the following Amazon review:
    "{review}"

    Please classify this review into one or more of the following Subject Level 3 categories. Provide your answer as a list of strings containing the EXACT labels from the options below.

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

                    # Validate response
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
                                f"Retrying SL3 classification for {sl1} > {sl2} (Attempt {attempt+1}/{self.max_retries})"
                            )
                    except (ValueError, SyntaxError):
                        print(
                            f"Retrying SL3 classification for {sl1} > {sl2} (Attempt {attempt+1}/{self.max_retries})"
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
Consider the following Amazon review:
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

                        # Validate response
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
                                    f"Retrying code assignment (Attempt {attempt+1}/{self.max_retries})"
                                )
                        except (ValueError, SyntaxError):
                            print(
                                f"Retrying code assignment (Attempt {attempt+1}/{self.max_retries})"
                            )
                    else:
                        raise ValueError(
                            f"Failed to assign codes after maximum retries"
                        )
        return subject4_results

    def extract_prior_product(self, review_text):
        prompt_template = PromptTemplate(
            input_variables=["review"],
            template="""
Consider the following Amazon review:
"{review}"

Determine if the consumer mentions a prior product they used before this one. If so, extract the brand and form (e.g., toothpaste, soap) mentioned by the consumer. Provide your answer as a string containing the brand and form. If no prior product is mentioned, return an empty string.

Your response MUST be a single string. Do not provide any additional text or explanations.

Example Outputs:
- "Crest toothpaste"
- "Dove soap"
- ""
""",
        )

        chain = LLMChain(prompt=prompt_template, llm=self.llm)

        for attempt in range(self.max_retries):
            with get_openai_callback() as cb:
                response = chain.run(review=review_text)
            # Accumulate token usage
            self.total_tokens += cb.total_tokens
            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens

            # Validate response
            response = response.strip().strip('"')
            if isinstance(response, str):
                # The response should be a string (possibly empty)
                return response
            else:
                print(
                    f"Retrying Prior Product extraction (Attempt {attempt+1}/{self.max_retries})"
                )
        return ""  # Return empty string if extraction fails after retries

    def determine_buy_again(self, review_text):
        prompt_template = PromptTemplate(
            input_variables=["review"],
            template="""
Consider the following Amazon review:
"{review}"

Determine if the consumer indicates whether they would purchase the product again. The possible values are:
- "Y" (Yes) if the consumer indicates they would purchase again.
- "N" (No) if the consumer indicates they would not purchase again.
- "D" (Don't Know) if the consumer says they don't know if they would purchase again.
- "" (empty string) if the consumer does not indicate.

Your response MUST be one of the following single characters: "Y", "N", "D", or "" (empty string). Do not provide any additional text or explanations.

Example Outputs:
- "Y"
- "N"
- "D"
- ""
""",
        )

        chain = LLMChain(prompt=prompt_template, llm=self.llm)

        for attempt in range(self.max_retries):
            with get_openai_callback() as cb:
                response = chain.run(review=review_text)
            # Accumulate token usage
            self.total_tokens += cb.total_tokens
            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens

            # Validate response
            response = response.strip().strip('"')
            if response in ["Y", "N", "D", ""]:
                return response
            else:
                print(
                    f"Retrying Buy Again determination (Attempt {attempt+1}/{self.max_retries})"
                )
        return ""  # Return empty string if determination fails after retries


def main():
    start_time = time.time()
    # Load the hierarchy JSON
    with open("subject_hierarchy_dict.json", "r", encoding="utf-8") as f:
        hierarchy = json.load(f)

    # Read the input CSV file
    input_csv = "input_reviews.csv"  # Replace with your actual input CSV file path
    df = pd.read_csv(input_csv)

    # Limit to first 5 rows for testing

    output_rows = []

    # Initialize total token counters
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    total_reviews = len(df)

    for idx, row in df.iterrows():
        review_text = f"""\
Title: {row["Title"]}
Description: {row["Review"]}

Product Tags: Brand={row["Brand"]}, Category={row["Product 1"]}, SubCategory={row["Product 1"]}, Product Variant={row["Product 1"]}
"""
        print(f"\nProcessing review {idx+1} of {total_reviews}")
        print(f"Review text:\n{review_text}")

        classifier = ReviewClassifier(hierarchy)
        try:
            subject4_list, prior_product, buy_again = classifier.classify_review(
                review_text
            )
            # Accumulate tokens
            total_prompt_tokens += classifier.prompt_tokens
            total_completion_tokens += classifier.completion_tokens
            total_tokens += classifier.total_tokens

            if not subject4_list:
                # No codes assigned
                continue

            # Print assigned subject codes
            print(f"Assigned Subject Codes: {subject4_list}")

            # For each code assigned, create a new row
            for subject4 in subject4_list:
                new_row = row.copy()
                new_row["Subject 4"] = subject4
                new_row["Prior Product"] = prior_product
                new_row["Buy Again"] = buy_again
                output_rows.append(new_row)
        except ValueError as e:
            print(f"Failed to classify review at index {idx}: {e}")
            # You can choose to handle the error as needed

    # Create a new DataFrame from output_rows
    if output_rows:
        output_df = pd.DataFrame(output_rows)
        # Save to CSV
        output_csv = "output12312.csv"  # Replace with your desired output file name
        output_df.to_csv(output_csv, index=False)
        print(f"Output saved to {output_csv}")
    else:
        print("No classifications were made.")

    print(f"Total input tokens (prompt tokens): {total_prompt_tokens}")
    print(f"Total output tokens (completion tokens): {total_completion_tokens}")
    print(f"Total tokens used: {total_tokens}")

    # Print total token usage
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    average_time_per_review = elapsed_time / total_reviews

    print(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds")
    print(f"Average time per review: {average_time_per_review:.2f} seconds")


if __name__ == "__main__":
    main()

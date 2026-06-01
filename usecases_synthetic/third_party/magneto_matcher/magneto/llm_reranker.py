import warnings
import json_repair
from litellm import completion

class LLMReranker:
    def __init__(self, llm_model="openai/gpt-4o-mini", **llm_model_kwargs):
        self.llm_model = llm_model
        self.llm_model_kwargs = llm_model_kwargs
        self.llm_attempts = 5


    def rematch(
        self,
        source_table,
        target_table,
        source_values,
        target_values,
        matched_columns,
        score_based=True,
    ):
        refined_matches = {}
        for source_col, target_col_scores in matched_columns.items():
            cand = (
                "Column: "
                + source_col
                + ", Sample values: ["
                + ",".join(source_values[source_col])
                + "]"
            )
            target_cols = [
                "Column: "
                + target_col
                + ", Sample values: ["
                + ",".join(target_values[target_col])
                + "]"
                for target_col, _ in target_col_scores
            ]
            targets = "\n".join(target_cols)
            attempts = 0
            while True:
                if attempts >= self.llm_attempts:
                    warnings.warn(
                        f"Failed to parse response after {self.llm_attempts} attempts. Skipping.",
                        UserWarning,
                    )
                    refined_match = []
                    for target_col, score in target_col_scores:
                        refined_match.append((target_col, score))
                    break
                refined_match = self._get_matches(cand, targets)
                refined_match = self._parse_matches(refined_match)
                attempts += 1
                if refined_match is not None:
                    break
            refined_matches[source_col] = refined_match
        return refined_matches

    def _get_prompt(self, cand, targets):
        prompt = (
            "Given a candidate column and a list of target columns, judge the similarity between the candidate and each target column. "
            "Return a JSON array of objects, each with 'column' (the target column name) and 'score' (a float between 0 and 1, two decimals, 1 is most similar). "
            "Do NOT provide any other output text or explanation. Only provide the JSON array.\n"
            "Example:\n"
            "Candidate Column: Column: EmployeeID, Sample values: [100, 101, 102]\n"
            "Target Schemas:\n"
            "Column: WorkerID, Sample values: [100, 101, 102]\n"
            "Column: EmpCode, Sample values: [001, 002, 003]\n"
            "Column: StaffName, Sample values: ['Alice', 'Bob', 'Charlie']\n"
            'Response: [\n  {"column": "WorkerID", "score": 0.95},\n  {"column": "EmpCode", "score": 0.30},\n  {"column": "StaffName", "score": 0.05}\n]\n\n'
            "Candidate Column: "
            + cand
            + "\n\nTarget Schemas:\n"
            + targets
            + "\n\nResponse: "
        )
        return prompt

    def _get_matches(self, cand, targets):
        prompt = self._get_prompt(cand, targets)
        messages = [
            {
                "role": "system",
                "content": "You are an AI trained to perform schema matching by providing column similarity scores.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        response = completion(
            model=self.llm_model,
            messages=messages,
            **self.llm_model_kwargs,
        )
        matches = response.choices[0].message.content
        return matches

    def _parse_matches(self, refined_match):
        try:
            matches_json = json_repair.loads(refined_match)
            matched_columns = []
            for entry in matches_json:
                schema_name = entry.get("column")
                score = float(entry.get("score", 0))
                matched_columns.append((schema_name, score))
            return matched_columns
        except Exception as e:
            warnings.warn(
                f"Error parsing JSON response: {e}\nRaw response: {refined_match}",
                UserWarning,
            )
            return None

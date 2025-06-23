import re

class Evaluator():
    def __init__(self) -> None:
        pass
    @staticmethod
    def find_answer_gsm8k(text):
        if "####" in text:
            return text.split("####")[-1].strip()
        return None

    @staticmethod
    def extract_predicted_answer(text, last=1):
        """
        Extract numerical or text answer from a response.
        
        Args:
            text: The text response to analyze
            last: int, number of numerical answers to return if \boxed{} not found.
                If 1, return the last one. If >1, return a tuple of the last N.
                
        Returns:
            str, tuple, or None: The extracted answer(s) or None if no valid answer found
        """
        # Try to find answer in \boxed{}
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}' 
        match = re.findall(pattern, text)
        if match:
            return match[-1]

        # If no \boxed{} in the text, extract numbers
        regex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore = [
            ",",
            r"\$",
            r"(?s).*#### ",
            r"\.$"
        ]
        matches = re.findall(regex_pattern, text)
        if matches:
            results = []
            for m in matches:
                val = [i for i in m if i][0].strip()
                for regex in regexes_to_ignore:
                    val = re.sub(regex, "", val)
                results.append(val)
            if last == 1:
                return results[-1]
            else:
                return tuple(results[-last:]) if len(results) >= last else tuple(results)
        else:
            return None

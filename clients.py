import openai
from concurrent.futures import ThreadPoolExecutor

# Define a mapping of model names to their corresponding prompt templates
MODEL_TEMPLATES = {
    model: lambda p: "<|User|>" + p + "<|Assistant|>"
    for model in [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-34B",
    ]
}

MODEL_TEMPLATES.update(
    {
        model: lambda p: (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            + p
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        for model in ["Qwen/Qwen2.5-Math-7B-Instruct"]
    }
)


def apply_chat_template(prompt, model_name):
    # Get the template function for the model, default to identity function
    template_fn = MODEL_TEMPLATES.get(
        model_name, lambda p: "<|User|>" + p + "<|Assistant|>"
    )
    return template_fn(prompt)


class ClientModel:
    def __init__(self, model_name, url, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=url,
        )
        self.model_name = model_name

    def generate(self, prompt, max_tokens=2048, temperature=0.6, top_p=0.95, n=1):
        completions = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            echo=False,
            n=n,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        return completions

    def prepare_prompt(self, prompt):
        pass


class vllmClientModel(ClientModel):
    def __init__(self, model_name, url, api_key):
        super().__init__(model_name, url, api_key)
        self.model_name = model_name
        self.url = url
        self.api_key = api_key
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_batch(
        self,
        prompts,
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        n=1,
        is_actives=None,
    ):
        def generate_completion(prompt, is_active, max_token):
            if is_active and max_token > 0:
                return self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    echo=False,
                    n=n,
                    stream=False,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_token,
                    logprobs=2,
                    seed=23, 
                )
            else:
                return None

        # Create a list to store completions in order
        completions = [None] * len(prompts)
        # Convert max_tokens to a list if it's not already
        if not isinstance(max_tokens, list):
            max_tokens = [max_tokens] * len(prompts)
        # Use ThreadPoolExecutor to parallelize requests while maintaining order
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and store futures with their indices
            future_to_idx = {
                executor.submit(generate_completion, prompt, is_active, max_token): idx
                for idx, (prompt, is_active, max_token) in enumerate(
                    zip(prompts, is_actives, max_tokens)
                )
            }

            # As futures complete, store results in correct positions
            for future in future_to_idx:
                idx = future_to_idx[future]
                completions[idx] = future.result()

        return completions

    def generate_batch_probe(
        self,
        prompts,
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        n=1,
        is_actives=None,
    ):
        return self.generate_batch(
            prompts, max_tokens, temperature, top_p, n, is_actives
        )

    def prepare_prompt(self, prompt):
        chat_template = apply_chat_template(prompt, self.model_name)
        # print(chat_template)
        return chat_template

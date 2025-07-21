import json
from openai import OpenAI
from tqdm import tqdm


class OpenAIWrapper:
    def __init__(self, api_key, base_url=None):
        if base_url is None:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.last_batch_input_file = None
    
    def prompt_to_msg(self, prompt):
        return [{"role": "user", "content": prompt}]

    def text_gen(self, messages, model_name="gpt-4o", ignore_gen_params=False, **kwargs):
        if isinstance(messages, str):
            messages = self.prompt_to_msg(messages)
        if ignore_gen_params:
            res = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs
            )
        else:
            res = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,  # Greedy decoding
                top_p=1,         # No nucleus sampling
                n=1,              # Single output
            )
        if len(res.choices) == 1:
            return res.choices[0].message.content
        else:
            return [res.choices[i].message.content for i in range(len(res.choices))]
    
    def embed(self, input, model="text-embedding-ada-002"):
        if isinstance(input, str):
            input = [input]
        res = self.client.embeddings.create(
            model=model,
            input=input
        )
        return [e.embedding for e in res.data]
    
    def generate_batch_file(self,conversations, model_name="gpt-4o", job_id="job-1"):
        requests = []
        file_name_w_path = "batch_requests/"+job_id+".jsonl"
        for i,messages in enumerate(conversations):
            request = {"custom_id": f"{job_id}-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": model_name,
                             "temperature": 0,  # Greedy decoding
                                "top_p": 1,         # No nucleus sampling
                                "n": 1,              # Single output
                                "messages": messages}}
            requests.append(request)

        with open(f'{file_name_w_path}', 'w') as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')

        batch_input_file = self.client.files.create(
            file=open(file_name_w_path, "rb"),
            purpose="batch"
            )
        
        self.last_batch_input_file = batch_input_file
        return batch_input_file
    
    def create_batch_job(self, batch_input_file_id=None, desc="job-1"):
        if batch_input_file_id is None:
            batch_input_file_id = self.last_batch_input_file.id
        
        create_resp = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": desc,
            }
        )
        self.last_create_resp = create_resp

        return create_resp
    
    def get_batch_job_results(self, create_resp_id=None):
        if create_resp_id is None:
            create_resp_id = self.last_create_resp.id
        batch_resp = self.client.batches.retrieve(create_resp_id)
        if batch_resp.status == "completed":
            out_file_id = batch_resp.output_file_id
            file_response = self.client.files.content(out_file_id)
            gpt_responses = file_response.text.split("\n")
            gpt_responses = [json.loads(resp)["response"]["body"]["choices"][0]["message"]["content"] for resp in gpt_responses if resp]
            return gpt_responses
        else:
            print("Job not completed, current status is:", batch_resp.status, "Completed:", batch_resp.request_counts.completed, "Total:", batch_resp.request_counts.total)
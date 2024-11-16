from concurrent.futures import ThreadPoolExecutor, wait

from openai import OpenAI
from together import Together
from tqdm import tqdm


class Batcher:
    def __init__(
        self,
        api_name,  # together/vllm
        api_key,
        model_name,
        temperature,
        system_prompt="",
        max_token=256,
        num_workers=64,
        timeout_duration=60,
        retry_attempts=2,
        api_base_url=None,
    ):

        if api_name == "vllm":
            self.client = OpenAI(api_key=api_key)
        elif api_name == "together":
            self.client = Together(api_key=api_key)
        else:
            raise ValueError("Invalid API name")
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index = []
        self.max_token = max_token
        if api_base_url:
            self.client.base_url = api_base_url

    def get_attitude(self, ask_text):
        index, ask_text = ask_text
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ask_text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_token,
            )
            return (index, completion.choices[0].message.content)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.miss_index.append(index)
            return (index, None)

    def process_attitude(self, message_list):
        new_list = []
        num_workers = self.num_workers
        timeout_duration = self.timeout_duration
        retry_attempts = 2

        executor = ThreadPoolExecutor(max_workers=num_workers)
        message_chunks = list(self.chunk_list(message_list, num_workers))
        try:
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {
                    executor.submit(self.get_attitude, message): message
                    for message in chunk
                }
                for _ in range(retry_attempts):
                    done, not_done = wait(
                        future_to_message.keys(), timeout=timeout_duration
                    )
                    for future in not_done:
                        future.cancel()
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    future_to_message = {
                        executor.submit(
                            self.get_attitude, future_to_message[future]
                        ): future
                        for future in not_done
                    }
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            executor.shutdown(wait=False)
            return new_list

    def complete_attitude_list(self, attitude_list, max_length):
        completed_list = []
        current_index = 0
        for item in attitude_list:
            index, value = item
            # Fill in missing indices
            while current_index < index:
                completed_list.append((current_index, None))
                current_index += 1
            # Add the current element from the list
            completed_list.append(item)
            current_index = index + 1
        while current_index < max_length:
            print("Filling in missing index", current_index)
            self.miss_index.append(current_index)
            completed_list.append((current_index, None))
            current_index += 1
        return completed_list

    def chunk_list(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def handle_message_list(self, message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list

    def get_miss_index(self):
        return self.miss_index

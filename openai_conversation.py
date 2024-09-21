import openai

class OpenAIConversationHandler():
    """
    A class to manage conversations with the OpenAI API.

    Attributes:
        api_key (str): The OpenAI API Key.
        model_name (str): The name of the model to use for responses.
        system_context (str): The system context to use for the conversation.
        temperature (float): The temperature to use when generating responses.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The nucleus sampling parameter
        frequency_penalty (float): The frequency penalty parameter
        presence_penalty (float): The presence penalty parameter
    """

    def __init__(
            self, 
            api_key: str, 
            model_name: str, 
            system_context: str,
            temperature: float = 1,
            max_tokens: int = 1024,
            top_p: float = 1,
            frequency_penalty: float = 0,
            presence_penalty: float = 0 
        ) -> None:
        openai.api_key = api_key
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.system_context = system_context

        self.messages = []
        self.message_index = -1
        self.add_message('system', system_context)

    def clear_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.messages = []
        self.message_index = -1
        self.add_message('system', self.system_context)

    def add_message_from_gpt_response(self, gpt_response: dict) -> None:
        """
        Add a message to the conversation from a GPT response.

        Args:
            gpt_response: The response from the GPT model.
        """
        reply = gpt_response.choices[0].message.content
        self.add_message('assistant', reply)

    def add_message_from_user(self, user_msg: str) -> None:
        """
        Add a user's message to the conversation.

        Args:
            user_msg (str): The user's message.
        
        Raises:
            ValueError: If the user message is not a string.
        """
        if type(user_msg) is not str:
            raise ValueError(f"Incorrect message type - expecting string, user_msg is {type(user_msg)}")
        self.add_message('user', user_msg)
        
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.

        Args:
            role (str): The role of the message sender ('system', 'user', or 'assistant').
            content (str): The content of the message.
        """
        self.messages.append({
            "role": role,
            "content": content
        })
        self.message_index += 1

    def get_chatgpt_response(self):
        """
        Get a response from the GPT model and add it to the conversation.
        
        Raises:
            openai.error.APIError: If an OpenAI API error occurs.
            Exception: If an unexpected error occurs.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature = self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            self.add_message_from_gpt_response(gpt_response=response)
        except openai.error.APIError as e:
            print(f"An OpenAI error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_response(self, index: int = None) -> str:
        """
        Get a response from the conversation history.

        Args:
            index (int): The index of the message to retrieve. If None, the most recent message is retrieved.
        
        Returns:
            The content of the message.
        """
        if index is None:
            index = self.message_index
        
        return self.messages[index].content
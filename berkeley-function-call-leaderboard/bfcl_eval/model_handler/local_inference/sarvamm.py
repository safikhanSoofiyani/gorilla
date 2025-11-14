from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class SarvamMHandler(OSSHandler):
    """
    Handler for the `sarvam_m` model family using its custom chat template
    with thinking enabled.

    Template (informal):
      BOS
      [SYSTEM_PROMPT]...[/SYSTEM_PROMPT]
      [INST]user_1[/INST]assistant_1
      [INST]user_2[/INST]assistant_2
      ...
      [INST]last_user[/INST]<think>\n   <-- model starts generating here
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        # explicitly forward dtype
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            registry_name=registry_name,
            is_fc_model=is_fc_model,
            dtype=dtype,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Prompt formatting (Python version of your Jinja chat template)
    # ---------------------------------------------------------------------
    @override
    def _format_prompt(self, messages, function) -> str:
        """
        Python equivalent of the sarvam_m chat template with thinking enabled.
        We ignore `enable_thinking` (always treated as True) and `function`
        (no tools/functions used in this template).
        
        {%- set today = strftime_now("%Y-%m-%d") %}

        {%- set default_think_system_message =
        "You are a helpful assistant. Think deeply before answering the user's question. \
        Do the thinking inside <think>...</think> tags."
        %}

        {%- set default_nothink_system_message =
        "You are a helpful assistant."
        %}

        {%- set think_system_message_addition =
        "Think deeply before answering the user's question. \
        Do the thinking inside <think>...</think> tags."
        %}

        {{- bos_token }}

        {%- if messages[0].role == 'system' %}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {%- set system_message = messages[0].content %}
            {%- else %}
                {%- set system_message = think_system_message_addition ~ '\n\n' ~ messages[0].content %}
            {%- endif %}
            {%- set loop_messages = messages[1:] %}
        {%- else %}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {%- set system_message = default_nothink_system_message %}
            {%- else %}
                {%- set system_message = default_think_system_message %}
            {%- endif %}
            {%- set loop_messages = messages %}
        {%- endif %}

        {{- '[SYSTEM_PROMPT]' ~ system_message ~ '[/SYSTEM_PROMPT]' }}

        {%- for message in loop_messages %}

            {%- if message.role == 'user' %}
                {%- if loop.index0 % 2 == 1 %}
                    {{- raise_exception('User and assistant turns must alternate starting with user turn!') }}
                {%- endif %}

                {%- if loop.last and not (enable_thinking is defined and enable_thinking is false) %}
                    {{- '[INST]' ~ message.content ~ '[/INST]<think>\n' }}
                {%- else %}
                    {{- '[INST]' ~ message.content ~ '[/INST]' }}
                {%- endif %}

            {%- elif message.role == 'system' %}
                {{- raise_exception('System message can only be the first message!') }}

            {%- elif message.role == 'assistant' %}
                {%- if loop.index0 % 2 == 0 %}
                    {{- raise_exception('User and assistant turns must alternate starting with user turn!') }}
                {%- endif %}

                {%- set content = message.content %}
                {%- set reasoning_content = '' %}

                {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in message.content %}
                        {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                        {%- set reasoning_content =
                            message.content.split('</think>')[0]
                                        .rstrip('\n')
                                        .split('<think>')[-1]
                                        .lstrip('\n')
                        %}
                    {%- endif %}
                {%- endif %}

                {%- if loop.last and reasoning_content and not (enable_thinking is defined and enable_thinking is false) %}
                    {{- '<think>\n'
                        ~ reasoning_content.strip('\n')
                        ~ '\n</think>\n\n'
                        ~ content.lstrip('\n')
                        ~ eos_token
                    }}
                {%- else %}
                    {{- content ~ eos_token }}
                {%- endif %}

            {%- else %}
                {{- raise_exception('Only user, system and assistant roles are supported!') }}
            {%- endif %}

        {%- endfor %}

        """

        # Get BOS/EOS tokens from tokenizer
        bos_token = self.tokenizer.bos_token or ""
        eos_token = self.tokenizer.eos_token or ""

        default_think_system_message = (
            "You are a helpful assistant. "
            "Think deeply before answering the user's question. "
            "Do the thinking inside <think>...</think> tags."
        )
        default_nothink_system_message = "You are a helpful assistant."
        think_system_message_addition = (
            "Think deeply before answering the user's question. "
            "Do the thinking inside <think>...</think> tags."
        )

        # --- Determine system_message and loop_messages ---
        # We always assume thinking is enabled, so we follow the "thinking" branches.
        if messages and messages[0]["role"] == "system":
            # enable_thinking != false => prepend the think addition
            system_message = (
                think_system_message_addition + "\n\n" + messages[0]["content"]
            )
            loop_messages = messages[1:]
        else:
            # no explicit system => use default "think" system prompt
            system_message = default_think_system_message
            loop_messages = messages

        # --- Start building the formatted prompt ---
        formatted_prompt = ""
        formatted_prompt += bos_token + "\n\n"
        formatted_prompt += f"[SYSTEM_PROMPT]{system_message}[/SYSTEM_PROMPT]"


        # --- Serialize loop_messages (user/assistant turns) ---
        # The template expects: user, assistant, user, assistant, ...
        # starting with user (index 0 in loop_messages).
        num_loop = len(loop_messages)

        for idx, message in enumerate(loop_messages):
            role = message["role"]
            content = message.get("content", "") or ""

            if role == "user":
                # Enforce alternation: user must be at even positions (0, 2, 4, ...)
                if idx % 2 == 1:
                    raise ValueError(
                        "User and assistant turns must alternate starting with user turn! "
                        f"Found user at loop index {idx}."
                    )

                is_last = idx == num_loop - 1

                if is_last:
                    # Last user turn: add <think>\n for thinking-enabled mode
                    formatted_prompt += f"[INST]{content}[/INST]<think>\n"
                else:
                    formatted_prompt += f"[INST]{content}[/INST]"

            elif role == "assistant":
                # Enforce alternation: assistant must be at odd positions (1, 3, 5, ...)
                if idx % 2 == 0:
                    raise ValueError(
                        "User and assistant turns must alternate starting with user turn! "
                        f"Found assistant at loop index {idx}."
                    )

                # Extract visible content + reasoning_content
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]
                elif "</think>" in content:
                    # Parse inline <think>...</think>
                    parts = content.split("</think>")
                    # Everything after </think> is visible content
                    content = parts[-1].lstrip("\n")
                    # Everything between <think> and </think> is reasoning
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )

                is_last = idx == num_loop - 1

                if is_last and reasoning_content:
                    # For the last assistant in history, if we have reasoning
                    # and thinking is enabled, re-materialize the think tags.
                    formatted_prompt += (
                        "<think>\n"
                        + reasoning_content.strip("\n")
                        + "\n</think>\n\n"
                        + content.lstrip("\n")
                        + eos_token
                    )
                else:
                    # Historical assistant or no reasoning: just append content + EOS
                    formatted_prompt += content + eos_token

            else:
                # The template only supports system (already handled as first),
                # user, and assistant.
                raise ValueError(
                    f"Only 'user', 'system', and 'assistant' roles are supported in "
                    f"sarvam_m template, but got: {role!r}"
                )

        return formatted_prompt

    # ---------------------------------------------------------------------
    # Response parsing: split <think>...</think> from visible answer
    # ---------------------------------------------------------------------
    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Split the model completion into reasoning (inside <think>...</think>)
        and the user-visible answer (after </think>).
        """
        model_response = api_response.choices[0].text

        reasoning_content = ""
        cleaned_response = model_response

        if "</think>" in model_response:
            parts = model_response.split("</think>")
            # Visible answer is everything after </think>
            cleaned_response = parts[-1].lstrip("\n")
            # Reasoning is everything between <think> and </think>
            reasoning_content = (
                parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            )

        # Optionally strip trailing EOS token if present
        eos_token = self.tokenizer.eos_token or ""
        if eos_token and cleaned_response.endswith(eos_token):
            cleaned_response = cleaned_response[: -len(eos_token)].rstrip()

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    # ---------------------------------------------------------------------
    # Store reasoning_content alongside content in conversation history
    # ---------------------------------------------------------------------
    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Save both visible content and reasoning_content so that future calls
        to `_format_prompt` can reconstruct <think> blocks if needed.
        """
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses"],
                "reasoning_content": model_response_data.get("reasoning_content", ""),
            }
        )
        return inference_data

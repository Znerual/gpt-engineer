import json

from typing import Any, Callable, Iterator, List, Optional, Union

import requests

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import TextGen
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _create_retry_decorator(
    llm: TextGen,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle exceptions"""

    return create_base_retry_decorator(
        error_types=[Exception], max_retries=3, run_manager=run_manager
    )


class CustomNonChatModel(BaseChatModel):
    llm: Optional[TextGen] = None

    def __init__(
        self,
        model_url,
        streaming: bool = False,
    ):
        super().__init__()
        if streaming:
            self.llm = TextGen(
                model_url=model_url,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
        else:
            self.llm = TextGen(model_url=model_url)

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(messages, stop, **kwargs: Any) -> Any:
            stream = kwargs.pop("stream", False)
            prompt_value = self._convert_input(messages)
            request = self.llm._get_parameters(stop=stop, **kwargs)
            request["prompt"] = prompt_value.to_string()
            request["stream"] = stream
            return requests.post(self.llm.model_url, json=request, stream=stream)

        return _completion_with_retry(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.completion_with_retry(
            messages=messages,
            run_manager=run_manager,
            stream=stream,
            stop=stop,
            **kwargs,
        )

        generations = []
        if stream:
            for chunk in response:
                if not isinstance(chunk, dict):
                    print(chunk)
                    chunk = json.loads(chunk)
                if len(chunk["choices"]) == 0:
                    continue
                choice = chunk["choices"][0]
                finish_reason = choice.get("finish_reason")
                generation_info = (
                    dict(finish_reason=finish_reason)
                    if finish_reason is not None
                    else None
                )
                gen = ChatGeneration(
                    AIMessage(content=choice.text), generation_info=generation_info
                )
                generations.append(gen)
                print(gen)
        else:
            if not isinstance(response, dict):
                response = response.json()

            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )

            message = AIMessage(content=choice["text"])
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
            print(gen)

        return ChatResult(generations=generations, llm_output=None)

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        # return self.llm(prompt=prompt_value.to_string(), stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self.completion_with_retry(
            messages=messages, run_manager=run_manager, stop=stop, stream=True, **kwargs
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )

            chunk = AIMessageChunk(content=choice.text, generation_info=generation_info)
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _llm_type(self):
        return "textgen"

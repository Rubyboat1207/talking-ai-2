from abc import ABCMeta
from typing import Iterable
from json import dumps, loads
from openai import Client, NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolParam

from agent import Agent, AgentResponse, HumanContext, EnvironmentalContext, ToolCallContext, AgentContext, \
    ToolCallResponseContext, FinishReason, SystemPromptContext


class OpenAiAgent(Agent, metaclass=ABCMeta):
    def __init__(self, speech_provider):
        super().__init__(speech_provider)
        self.client = Client()

    def generate_response(self) -> AgentResponse:
        messages: list[ChatCompletionMessageParam] = []

        last_agent_ctx: ChatCompletionAssistantMessageParam | None = None

        for entry in self._ctx:
            if isinstance(entry, HumanContext):
                messages.append({"role": "user", "content": entry.value})
            elif isinstance(entry, EnvironmentalContext):
                messages.append({"role": "system", "content": f"environmental context: {entry.value}"})
            elif isinstance(entry, ToolCallContext):
                tc = {
                    'id': entry.guid,
                    'type': 'function',
                    'function': {
                        'name': entry.value,
                        'arguments': dumps(entry.parameters)
                    }
                }

                if last_agent_ctx.get('tool_calls') is None:
                    last_agent_ctx['tool_calls'] = [tc]
                else:
                    # assume this is a list, because we literally just made it
                    last_agent_ctx['tool_calls'].append(tc)

                if entry.response_id is None:
                    messages.append({
                        'role': 'tool',
                        'content': 'response is async, and is currently working. You will receive a result soon.',
                        'tool_call_id': entry.guid
                    })
            elif isinstance(entry, ToolCallResponseContext):
                messages.append({
                    'role': 'tool',
                    'content': entry.value,
                    'tool_call_id': entry.call_id
                })
            elif isinstance(entry, AgentContext):
                messages.append({
                    'role': 'assistant',
                    'content': entry.value
                })
                last_agent_ctx = messages[len(messages) - 1]
            elif isinstance(entry, SystemPromptContext):
                messages.append({
                    'role': 'system',
                    'content': entry.value
                })

        tools: list[ChatCompletionToolParam] = []

        for action in self.action_manager.actions.values():
            tools.append({
                'function': {
                    'name': action.name,
                    'description': action.description,
                    'parameters': action.parameter_schema
                },
                'type': 'function'
            })
        for group in self.action_manager.ephemeral_groups.values():
            for action in group:
                tools.append({
                    'function': {
                        'name': action.name,
                        'description': action.description,
                        'parameters': action.parameter_schema
                    },
                    'type': 'function'
                })

        print(dumps(messages, indent=2))

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools if len(tools) > 0 else NOT_GIVEN,
            tool_choice=NOT_GIVEN if len(tools) == 0 else 'auto' if
            len(self.action_manager.forced_actions_queue) == 0
            else {'function': {'name': self.action_manager.forced_actions_queue[0]}, 'type': 'function'}
        )

        choice = response.choices[0]

        finish_reason = FinishReason.STOP

        print(choice.model_dump_json())

        if choice.finish_reason == "tool_calls":
            finish_reason = FinishReason.TOOL_CALL

        tcs: list[ToolCallContext] | None = None

        if choice.message.tool_calls is not None:
            tcs = []

            for tool_call in choice.message.tool_calls:
                tcs.append(ToolCallContext(tool_call.function.name, loads(tool_call.function.arguments)))

        agent_res = AgentResponse(choice.message.content, tcs, finish_reason)

        if not self.action_manager.response_meets_action_criteria(agent_res):
            print('agent did not respond to criteria')
            agent_res = self.generate_response()

        return agent_res

from dotenv import load_dotenv
from agent import HumanContext, Action, FinishReason
from agents.openai_agent import OpenAiAgent
from speech_providers.styletts2_speech_provider import StyleTTS2SpeechProvider

load_dotenv()

agent = OpenAiAgent(StyleTTS2SpeechProvider())


def pr(val: dict[str, ...]):
    print(val['output'])
    return 'OK'


print_to_console = Action(
    {'type': 'object', 'properties': {'output': {'type': 'string', 'description': 'the console output'}}},
    'prints the output to the standard output, only print when asked.', 'print', pr)

agent.action_manager.register_action(print_to_console)

while True:
    agent.add_context(HumanContext(input('speak to it: ')))

    res = None

    while res is None or res.finish_reason != FinishReason.STOP:
        res = agent.generate_response()

        agent.add_response_to_context(res, True)

    agent.speak_recent_response()

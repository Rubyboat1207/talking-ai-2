from threading import Thread

from dotenv import load_dotenv
from agent import HumanContext, Action, FinishReason
from agents.openai_agent import OpenAiAgent
from speech_providers.console_output_speech_provider import ConsoleOutputSpeechProvider
from websocket import WebsocketManager
import asyncio

load_dotenv()

agent = OpenAiAgent(ConsoleOutputSpeechProvider())

async def main():
    asyncio.create_task(WebsocketManager(agent).init_websocket())

    loop = asyncio.get_running_loop()

    while True:
        user_input = await loop.run_in_executor(None, input, 'speak to it: ')
        agent.add_context(HumanContext(user_input))

        res = None

        while res is None or res.finish_reason != FinishReason.STOP:
            res = agent.generate_response()

            await agent.add_response_to_context(res, True)

        agent.speak_recent_response()

asyncio.run(main())
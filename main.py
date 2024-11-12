import asyncio
from threading import Thread
from dotenv import load_dotenv
from agent import HumanContext, Action, FinishReason
from agents.openai_agent import OpenAiAgent
from speech_providers.styletts2_speech_provider import StyleTTS2SpeechProvider
from stt import stt
from websocket import WebsocketManager

load_dotenv()

agent = OpenAiAgent(StyleTTS2SpeechProvider())


async def main():
    websocket_manager = WebsocketManager(agent)

    # Create the websocket task and await it in the background
    websocket_task = asyncio.create_task(websocket_manager.init_websocket())

    loop = asyncio.get_running_loop()

    try:
        while True:
            # Run blocking input in an executor to avoid blocking the event loop
            user_input = await stt()
            agent.add_context(HumanContext(user_input))

            res = None
            while res is None or res.finish_reason != FinishReason.STOP:
                res = agent.generate_response()
                await agent.add_response_to_context(res, True)

            agent.speak_recent_response()

    except asyncio.CancelledError:
        print("Main function cancelled, shutting down...")

    finally:
        # Ensure websocket_task is awaited and handled
        websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            print("Websocket task cancelled and shut down gracefully.")


asyncio.run(main())

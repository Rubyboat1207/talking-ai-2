import asyncio
import time
from threading import Thread
from dotenv import load_dotenv
from agent import HumanContext, FinishReason, SystemPromptContext
from agents.openai_agent import OpenAiAgent
# from speech_providers.styletts2_speech_provider import StyleTTS2SpeechProvider as Sp
from speech_providers.console_output_speech_provider import ConsoleOutputSpeechProvider as Sp
from stt import stt
from neuro_api_ws import NeuroSamaWebsocketManager

load_dotenv()

agent = OpenAiAgent(Sp())

# agent.context_added_notifiers.append(on_context_added)

agent.add_context(SystemPromptContext(
    "You are a TTS ai. Keep responses speakable and short. Dont make lists. Be decisive. No markdown is allowed."))


async def main():
    global on_env_ctx_added
    use_stt = False
    auto_prompt = False
    websocket_manager = NeuroSamaWebsocketManager(agent)

    websocket_task = asyncio.create_task(websocket_manager.init_websocket())

    loop = asyncio.get_running_loop()

    try:
        while True:
            if not auto_prompt:
                if use_stt:
                    text = await stt()
                else:
                    text = await loop.run_in_executor(None, input, 'speak to it: ')
                agent.add_context(HumanContext(text))

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

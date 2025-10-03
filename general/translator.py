from googletrans import Translator
import asyncio
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

translator = Translator()

async def _translate_async(text, src=None, dest='en'):
    """Async helper function for translation"""
    if src:
        result = await translator.translate(text, src=src, dest=dest)
    else:
        result = await translator.translate(text, dest=dest)
    return result.text

def _run_async(coro):
    """Run async function in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop, we need to use a different approach
            import concurrent.futures
            import threading

            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

def from_english_to_russian(text):
    try:
        return _run_async(_translate_async(text, src='en', dest='ru'))
    except Exception as e:
        print(f"Error translating to Russian: {e}")
        return None

def from_russian_to_english(text):
    try:
        return _run_async(_translate_async(text, src='ru', dest='en'))
    except Exception as e:
        print(f"Error translating to English: {e}")
        return None

if __name__ == '__main__':
    print(from_english_to_russian("Hello, world!"))
    print(from_russian_to_english("Привет, мир!"))

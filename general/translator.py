from deep_translator import GoogleTranslator

def from_english_to_russian(text):
    return GoogleTranslator(source='en', target='ru').translate(text=text)

def from_russian_to_english(text):
    return GoogleTranslator(source='ru', target='en').translate(text=text)

if __name__ == '__main__':
    print(from_english_to_russian("Hello, world!"))
    print(from_russian_to_english("Привет, мир!"))

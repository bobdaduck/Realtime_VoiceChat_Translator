# RT_Translator

In order to play games with chinese people who do not speak english, this python app provides a screen overlay with realtime pinyin and english translations. Audio output from the computer is captured and translated, with pinyin provided over the english translation in order to learn words and verify accuracy, and spoken english will be translated into pinyin, allowing rudimentary communication to be returned. 

# Process

Coded with the help of Artificial Intelligence, namely Claude, Deepseek, Copilot, and ChatGPT. Each helping with different aspects of the project. 

# Usage

*A vosk english model will need to be manually downloaded to the project directory as "vosk-en", and is not provided.* The chinese paraformer model will be downloaded automatically. This app uses these models, run locally, to transcribe spoken audio into english and chinese, and then those characters are sent to Baidu's API for translation. Baidu is best in class for zh-en translation tasks, and has higher accuracy than locally run models. Deepseek AI renders comparable translation quality but their API does cost money.
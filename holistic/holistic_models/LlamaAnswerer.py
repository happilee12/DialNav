from interface.AnswerGeneration import AnswerGeneration
from modules.qa.Llama.llama_answerer import LlamaAnswerer

class LlamaAnswerer(AnswerGeneration): 
    def __init__(self, basepath):
        caption_path = f"{basepath}/captions/llava_naive.json"
        self.llama_answerer = LlamaAnswerer(caption_path)

    def ask(self, scanIds, viewpoints, goals):
        natural_language_output = self.llama_answerer.say(scanIds, viewpoints, goals)
        return natural_language_output
    
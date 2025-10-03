from interface.QuestionGeneration import QuestionGeneration
from modules.qa.Llava.llava_questioner import LLAVAQuesetioner

class LlavaQuestioner(QuestionGeneration):
    def __init__(self, basepath):
        caption_path = f"{basepath}/captions/llava_naive.json"
        self.llava_questioner = LLAVAQuesetioner(caption_path)

    def ask(self, scanIds, viewpoints, goals):
        natural_language_output = self.llava_questioner.say(scanIds, viewpoints, goals)
        return natural_language_output
    
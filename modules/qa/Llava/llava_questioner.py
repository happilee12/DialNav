import json

class LLAVAQuesetioner():
    def __init__(self, caption_path):
        self.caption_list = None
        with open(caption_path) as f:
            self.caption_list = json.load(f)
    
    def say(self, scanIds, viewpoints, goals):
        result = []
        seen_paths = [[vp] for vp in viewpoints]
        for idx, vp in enumerate(viewpoints):
            try:
                caption = self.caption_list[scanIds[idx]][vp]
                if not caption:
                    caption = 'where should i go?'
                result.append(caption)
            except Exception as e:
                print(e)
                result.append('where should i go?')
        return result, seen_paths
        
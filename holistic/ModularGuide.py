from interface.Navigation import Navigation
from interface.WTA import WTA
from interface.AnswerGeneration  import AnswerGeneration
from interface.Localization import Localization
from interface.GuideAgent import GuideAgent
import numpy as np

class ModularGuide(GuideAgent):
    def __init__(self, args, answer_model: AnswerGeneration, localization_model: Localization, env_infos: dict):
        super().__init__(args, answer_model, localization_model)
        self.args = args
        self.answer_model = answer_model
        self.localization_model = localization_model

        self.shortest_distances = env_infos['shortest_distances']
        self.shortest_paths = env_infos['shortest_paths']
    
    def _choose_path(self, scanId, viewpoint, goal_list):
        distances_all = []
        paths_all = []
        for g in goal_list:
            distances_all.append(self.shortest_distances[scanId][viewpoint][g])
            paths_all.append(self.shortest_paths[scanId][viewpoint][g])
        sorted_indices = np.argsort(distances_all)
        sorted_paths = [paths_all[i] for i in sorted_indices]
        sorted_distances = [distances_all[i] for i in sorted_indices]
        path = sorted_paths[0]
        return path

    ##### Answer Functions #####
    def answer(self, *args, **kwargs):
        if self.answer_model is None:
            raise ValueError("answer_model is not set")
        return self.answer_model.answer(*args, **kwargs)
    
    ##### Localization Functions #####
    def localize(self, scanIds, questions):
        if self.localization_model is None:
            raise ValueError("localization_model is not set")
        return self.localization_model.localize(scanIds, questions)
    

class RandomLocalizationModule:
    def __init__(self, connectivity_dir):
        with open(connectivity_dir, "r") as file:
            self.vp_all = json.load(file)

    def localize(self, scan):
        localized = []
        for vp in vp_list:
            vp_caption = self.vp_all[vp]
            vps = list(vp_caption.keys())
            localized.append(random.choice(vps))
        return localized
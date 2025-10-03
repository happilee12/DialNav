import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class LlamaAnswerer():
    def __init__(self, caption_path, model_cache_dir):
        self.caption_list = None
        # with open('/home/master/00_WorkDir/04_VDNRA/90_datasets/captions/llava_captions.json') as f:
        with open(caption_path) as f:
            self.caption_list = json.load(f)
        self.llama = self.init_llama(model_cache_dir)

    def init_llama(self, model_cache_dir):
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",  cache_dir=model_cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=model_cache_dir,
            padding_side='left'  # decoder-only 모델을 위한 left padding 설정
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = model.config.eos_token_id

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        return pipe
    
    def get_prompt(self, current_description):
        prompt = "you are an agent for creating navigation route. Given sequence of scene image description, create a navigation guide sentence for the route. You don't have to describe every single step. Try to add any unique object or landmark. \n Return your evaluation results in the following JSON format without any additional text: \n {'response': '<your response>'}"
        # scan_desc = 'Scene 1: The lounge in this image is a small, cozy space with a couch and a chair. It is located in a hotel or a similar establishment, and it is designed for relaxation and comfort. The lounge is situated near a bathroom, which suggests that it is part of a larger living area or a suite. The presence of a potted plant in the room adds a touch of greenery and freshness to the space \nScene 2: The lounge in this image is a large, comfortable space with a couch and a chair. It is located near a large window, which allows natural light to fill the room. The lounge is part of a spacious living area that also includes a family room and a living room. The overall atmosphere of the lounge is inviting and relaxing, making it an ideal space for people to unwind and enjoy their time. \nScene 3: In the image, there is a lounge area with a couch and a chair. The couch is located on the right side of the room, while the chair is situated closer to the center. The lounge is situated near a doorway, which suggests that it is a comfortable space for people to relax and socialize. \nScene 4: The spa/sauna in the image is a large, open room with a blue-lit atmosphere. The room features a large bathtub, which is situated in the middle of the space. There are also two benches in the room, one on the left side and another on the right side. The overall design of the room is modern and inviting, making it an ideal space for relaxation and rejuvenation.'
        # sample_output = '{"response": "if you place the swimming pool to your left,you will see a narrow hallway that leads to a small room, and a brown door that leads to a medium sized room, with something like a stone bed in the middle. the room with the stone bed is the goal room."}'
        
        scan_desc = (
            'Scene 1: The bedroom in the image is a large, well-lit space with a clean and elegant design. It features a large bed situated near a window, providing a view of the outdoors. The room also contains a chair, a dresser, and a mirror. The bedroom is decorated with various potted plants, adding a touch of greenery and life to the space. The overall atmosphere of the room is inviting and comfortable. \n' + 'Scene 2: The hallway in this image is a large, open space with a wooden floor. It features a staircase leading to the upper floor, and a doorway that leads to the living room. The hallway is well-lit, with a light on the ceiling, and it has a clean and elegant appearance. \n' + 'Scene 3: The hallway in this image is spacious and well-lit, featuring a hardwood floor. It leads to a bedroom and a living room, creating a seamless flow between the spaces. The hallway is decorated with a few potted plants, adding a touch of greenery to the room. The overall atmosphere of the hallway is inviting and comfortable. \n' + 'Scene 4: The hallway in this image is a large, open space with a hardwood floor. It features a staircase leading to the upper floor, and a doorway that leads to a living room. The hallway is well-lit, with lights on the ceiling and a chandelier hanging above the staircase. The room also has a couch and a dog, adding a cozy and welcoming atmosphere to the space. \n' + 'Scene 5: The bedroom in the image is a large, well-lit space with a hardwood floor. It features a large bed, a chair, and a couch, providing ample seating options. The room also has a TV, a potted plant, and a vase, adding to the cozy and inviting atmosphere. The bedroom is connected to a living room, creating a spacious and open living area. \n' + 'Scene 6: The bedroom in the image is a cozy and well-organized space. It features a large bookshelf filled with various books, creating a comfortable and intellectual atmosphere. The room also has a fireplace, which adds warmth and charm to the space. A doorway leads to another room, and a staircase can be seen in the background. The room is decorated with a rug, and a potted plant is placed near the doorway, adding a touch of greenery to the room. \n' + 'Scene 7: The bathroom in the image is small and features a toilet and a sink. The toilet is located on the left side of the bathroom, while the sink is situated on the right side. The bathroom also has a door, which is open in the image. \n'
        )
        sample_output = '{"response": "Leave the bedroom, walk straight past the stairs to your left, and straight into the door in front of you. The door is nearest to a display with flowers and two silver vases. Walk straight until the end of the bedroom, and the bathroom on your left is the target."}'

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": scan_desc},
            {"role": "assistant", "content": sample_output}, 
            {"role": "user", "content": current_description}
        ]

    def get_path_caption(self, scan, path):
        try:
            text = ''
            for idx, vp in enumerate(path):
                caption = self.caption_list[scan][vp]
                description = f"Scene {idx+1}: {caption}"
                text += description+"\n" 
            return text
        except Exception as e:
            print(e)
            return 'no message'
    

    def get_answer(self, messages):
        def process_outputs(outputs, result, failed_idx):
            for idx, output_item in enumerate(outputs):
                if len(output_item) < 1:
                    raise ValueError("Parsed output JSON does not contain required fields.")
                result_content_str = output_item[0].get("generated_text")[-1].get('content', '{}')

                if result_content_str.startswith("assistant\n\n"):
                    result_content_str = result_content_str[len("assistant\n\n"):]
                
                try:
                    result_content = json.loads(result_content_str)
                    if not isinstance(result_content, dict) or "response" not in result_content:
                        failed_idx.append(idx)
                    else:
                        result_content['idx'] = idx
                        result.append(result_content)
                except Exception as e:
                    print(e, result_content_str)
                    failed_idx.append(idx)

            return result, failed_idx

        outputs = self.llama(messages, batch_size=len(messages))
        if not outputs or not isinstance(outputs, list):
            raise ValueError("Unexpected output format from pipe.")
        
        result = []
        failed_idx = []
        
        # 첫 번째 시도에서 결과 처리
        result, failed_idx = process_outputs(outputs, result, failed_idx)

        # 실패한 인덱스에 대해 재시도
        while failed_idx:
            print("retrying ... ", failed_idx)
            retry_messages = [messages[i] for i in failed_idx]
            retry_outputs = self.llama(retry_messages, batch_size=len(retry_messages))
            
            # 실패한 메시지에 대해 다시 처리
            new_result = []
            new_failed_idx = []
            new_result, new_failed_idx = process_outputs(retry_outputs, new_result, new_failed_idx)

            # 재시도 성공한 결과를 원래 인덱스에 맞춰 삽입
            for res in new_result:
                original_idx = failed_idx[res['idx']]
                res['idx'] = original_idx  # 원래 인덱스를 저장
                result.append(res)

            # 남은 실패 인덱스 갱신
            failed_idx = [failed_idx[i] for i in new_failed_idx]

        # 인덱스 순서대로 정렬
        result_sorted = sorted(result, key=lambda x: x['idx'])

        # 정렬된 응답만 반환
        result = [res['response'] for res in result_sorted] 
        return result


    # def say(self, env, caption_type='answer', given=[]):
    #     result = []
    #     obs = env._get_obs(t=0)
    #     viewpoints = [ob['viewpoint'] for ob in obs]
    #     # print("viewpoints", viewpoints)
    #     teacher_paths = [ob['teacher_path'] for ob in obs]
    #     # print("teacher_path", teacher_paths)
    #     scans = [ob['scan'] for ob in obs]

    #     path_captions = [self.get_path_caption(scan, path) for (scan, path) in zip(scans, teacher_paths)]
    #     # print("path captions : ", path_captions)
    #     messages = [self.get_prompt(caption) for caption in path_captions]
    #     result = self.get_answer(messages)
    #     return [], result, teacher_paths 
        

    def say(self, scans, teacher_paths):
        result = []
        path_captions = [self.get_path_caption(scan, path) for (scan, path) in zip(scans, teacher_paths)]
        messages = [self.get_prompt(caption) for caption in path_captions]
        result = self.get_answer(messages)
        return result, teacher_paths 
        

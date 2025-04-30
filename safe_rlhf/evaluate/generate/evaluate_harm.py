from HarmfulnessRewardModel import AbsoluteHarmfulnessPredictor, ConversationBuilder
import numpy as np
import json
text_folder = 'outputs/iboero16/SFT-BASE/' # home/panda/pda/pd-alignment/output/sft-safe-v2/run-20250429-154703
text_folder = 'outputs/home/panda/pda/pd-alignment/output/sft-safe-v2/run-20250429-154703/'
text_file = 'I-PhysicalSafetyUnsafe_eval'
text_path =  text_folder + text_file + '.json'
data = json.load(open(text_path, 'r'))

user_texts = data['instructions']
assistant_texts = data['outputs']
assistant_texts = [assistant_text.split('</s>')[0] for assistant_text in assistant_texts]

setup = "redteam-osst"  # or "redteam-osst"
harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)
harmfulness_scores = np.array(harmfulness_scores)
breakpoint()
# Save the scores numpy array
np.save(text_folder + text_file + '_scores.npy', harmfulness_scores)

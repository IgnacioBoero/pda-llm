from HarmfulnessRewardModel import AbsoluteHarmfulnessPredictor, ConversationBuilder

c

setup = "redteam"  # or "redteam-osst"
harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)

print(harmfulness_scores)
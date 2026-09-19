from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

def test():
    model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    
    claim = "St. Francis Institute of Technology is located in Mumbai."
    evidence = "St. Francis Institute of Technology (SFIT) in Mumbai, India, is an engineering college named after Francis of Assisi."
    
    # 0 = Contradiction, 1 = Entailment, 2 = Neutral
    print("Testing (Claim, Evidence)")
    scores = model.predict([(claim, evidence)])[0]
    print(f"Logits: {scores}")
    print(f"Argmax: {np.argmax(scores)}")
    
    print("\nTesting (Evidence, Claim)")
    scores2 = model.predict([(evidence, claim)])[0]
    print(f"Logits: {scores2}")
    print(f"Argmax: {np.argmax(scores2)}")

if __name__ == "__main__":
    test()

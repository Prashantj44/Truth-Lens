from typing import List, Dict, Any
import numpy as np

class NLIEvaluator:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLIEvaluator, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is None:
            import time
            from sentence_transformers.cross_encoder import CrossEncoder
            
            for attempt in range(3):
                try:
                    print(f"[NLIEvaluator] Loading NLI model (Attempt {attempt+1}/3)...")
                    self._model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
                    print("[NLIEvaluator] NLI model loaded successfully.")
                    return
                except Exception as e:
                    print(f"[NLIEvaluator] Error loading NLI model: {e}")
                    time.sleep(2)
            
            print("[NLIEvaluator] Failed to load NLI model after 3 attempts.")
            self._model = "fallback"

    def evaluate_stance(self, claim: str, evidence_text: str) -> Dict[str, Any]:
        """
        Evaluates the stance of evidence_text towards the claim using the NLI model.
        Returns: Dict containing 'stance' (SUPPORTING, CONTRADICTING, NEUTRAL) and 'score'.
        """
        if not evidence_text or len(evidence_text.strip()) < 10:
            return {"stance": "NEUTRAL", "score": 0.0, "rationale": "Evidence text too short."}

        self._load_model()

        if self._model == "fallback":
            # Fallback to neutral if model fails to load
            return {"stance": "NEUTRAL", "score": 0.5, "rationale": "Fallback mode active."}

        try:
            # NLI model requires (Premise, Hypothesis)
            # Premise = Evidence Text
            # Hypothesis = Claim
            scores = self._model.predict([(evidence_text, claim)])[0]
            
            # Convert logits to probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # The exact label mapping for cross-encoder/nli-deberta-v3-base:
            # 0: contradiction, 1: entailment, 2: neutral
            pred_idx = np.argmax(probs)
            max_prob = float(probs[pred_idx])

            if pred_idx == 0:
                stance = "CONTRADICTING"
                rationale = f"NLI model detected contradiction with {max_prob*100:.1f}% probability."
            elif pred_idx == 1:
                stance = "SUPPORTING"
                rationale = f"NLI model detected entailment with {max_prob*100:.1f}% probability."
            else:
                stance = "NEUTRAL"
                rationale = f"NLI model detected neutral context with {max_prob*100:.1f}% probability."

            # If confidence is too low, fallback to neutral
            if max_prob < 0.6 and stance != "NEUTRAL":
                stance = "NEUTRAL"
                rationale += " (Confidence too low, demoted to NEUTRAL)"

            return {
                "stance": stance,
                "score": max_prob,
                "rationale": rationale,
                "raw_probs": probs.tolist()
            }
        except Exception as e:
            print(f"[NLIEvaluator] NLI inference failed: {e}")
            return {"stance": "NEUTRAL", "score": 0.0, "rationale": f"Inference Error: {e}"}

nli_evaluator = NLIEvaluator()

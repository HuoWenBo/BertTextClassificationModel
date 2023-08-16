from typing import Literal, List, Dict


class BertTextClassificationModel:
    def __init__(
            self,
            *,
            classes: List[str] = None,
            device: Literal[0, -1] = -1
    ):
        from transformers import pipeline
        from transformers.pipelines import Pipeline

        classifier: Pipeline = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=device
        )
        self._classifier = classifier
        self._classes = classes

    def set_classes(self, classes: List[str]):
        self._classes = classes

    def predictive_classification(self, text: str, classes: List[str] = None) -> zip:
        if not (classes or self._classes):
            raise ValueError
        if not classes:
            classes = self._classes
        output = self._classifier(
            text,
            classes,
            multi_label=True
        )
        return zip(output.get('labels', []), output.get('scores', []))
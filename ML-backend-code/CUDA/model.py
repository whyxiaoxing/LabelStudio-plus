import logging
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from Pipeline import run_task

logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Run one image through the inference pipeline and return its prediction.

        Label Studio feeds tasks one at a time, so a request carries a single
        task: the image goes receive -> infer -> postprocess -> return and the
        result is sent back immediately (no batch accumulation).  The
        annotation type (box / mask) is detected automatically from the Label
        Studio label config, see Pipeline.detect_controls.

            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        if not tasks:
            logger.warning("No tasks received, project ID = %s", self.project_id)
            return ModelResponse(predictions=[])
        if len(tasks) > 1:
            logger.warning(
                "Received %s tasks in one request; this backend handles one "
                "image per request (Label Studio sends tasks one at a time), "
                "processing the first one only",
                len(tasks),
            )
        logger.info(
            "Running prediction for task %s, project ID = %s",
            tasks[0].get("id"),
            self.project_id,
        )
        prediction = run_task(self, tasks[0])
        return ModelResponse(predictions=[prediction])
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')


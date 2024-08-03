import base64
import os
from openai import AzureOpenAI

class GPT:
    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key="6d1ec23e65c94a65adca014b3548feee",
            api_version="2024-02-01",
            azure_endpoint="https://nsg-gpt4-vision-instance.openai.azure.com/"
        )
    def predict(self,image_path):
        base64_image = self.encode_image(image_path)

        deployment_name = 'gpt-4o'  # This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment.

        response = self.client.chat.completions.create(
            model = deployment_name,
            messages=[
                {"role": "system", "content": "You need to classify the images uploaded by the user as digits from 0-9"},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]},
                {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            ]
        )

        return response.choices[0].message.content

    def encode_image(self,image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Path to your image

    # Getting the base64 string
    
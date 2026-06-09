from roboflow import Roboflow

rf = Roboflow(api_key="WBy7jG6AiiqjzifOfiNH")
project = rf.workspace().project("dey022")
model = project.version(1).model

# infer on a local image
print(model.predict("INPUT_unseen_example.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("INPUT_unseen_example.jpg", confidence=40, overlap=30).save("OUTPUT_unseen_prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

from roboflow import Roboflow

rf = Roboflow(api_key="WBy7jG6AiiqjzifOfiNH")
# project = rf.workspace().project("card-nfetv")
# project = rf.workspace().project("karts-vc5wx")
project = rf.workspace().project("carddetection-v1hqz")
model = project.version(17).model

# infer on a local image
print(model.predict("INPUT_dealer_area_current_view.jpg", confidence=70, overlap=100).json())

# visualize your prediction
model.predict("INPUT_dealer_area_current_view.jpg", confidence=70, overlap=100).save("OUTPUT_dealer_prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

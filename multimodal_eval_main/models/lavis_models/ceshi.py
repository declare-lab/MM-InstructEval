from lavis.models import load_model
from lavis.models import load_model_and_preprocess
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2", 
                                  model_type="pretrain", 
                                  is_eval=True,
                                  device=device)
# model.name: Blip2Qformer
text = "did you know this couch can turned into fully fledged comfortable bed jannahhotels"
image_path = '/data/xiaocui/data/Multimodal_Classification/Multimodal_Sentiment_Classification/MVSA_Single/image_data/test_image/1092.jpg'
raw_image = Image.open(image_path).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](text)
sample = {"image": image, "text_input": [text_input]}

features_multimodal = model.extract_features(sample)
print(features_multimodal.multimodal_embeds.shape)
# torch.Size([1, 32, 768]), use features_multimodal[:,0,:] for multimodal classification tasks

features_image = model.extract_features(sample, mode="image")
features_text = model.extract_features(sample, mode="text")
print(features_image.image_embeds.shape)
# torch.Size([1, 32, 768])
print(features_text.text_embeds.shape)
# torch.Size([1, 16, 768])

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 32, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 16, 768])

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 32, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 16, 256])
similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
print(similarity)